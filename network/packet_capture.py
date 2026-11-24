import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from scapy.all import sniff, PcapReader, IP, IPv6, TCP, UDP, conf
from config import TSHARK_DEFAULT_PATHS
from network.flow_manager import FlowManager
from network.flow_extractor import FlowRecord
from utils.logger import logger

class PacketCapture:
    """Captures live network packets or reads offline PCAP/PCAPNG files into network flows."""

    CAPTURE_POLL_INTERVAL_SECONDS = 1.0

    def __init__(self, flow_timeout: float = 60.0):
        self.flow_manager = FlowManager(timeout=flow_timeout)
        self.is_capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        self.packet_count = 0
        self.capture_error: Optional[str] = None
        self.completed_flows_queue: queue.Queue = queue.Queue()

    @staticmethod
    def list_interfaces() -> List[Dict[str, Any]]:
        """Discovers available network interfaces on Windows."""
        interfaces = []
        try:
            ifaces = conf.ifaces
            for iface_name, iface_obj in ifaces.items():
                desc = getattr(iface_obj, "description", str(iface_name))
                ip = getattr(iface_obj, "ip", "")
                interfaces.append({
                    "name": str(iface_name),
                    "description": str(desc),
                    "ip": str(ip)
                })
        except Exception as e:
            logger.error(f"Error enumerating interfaces: {e}")
        return sorted(
            interfaces,
            key=lambda interface: (
                not interface["ip"]
                or interface["ip"].startswith(("169.254.", "127."))
                or interface["ip"] == "0.0.0.0",
                interface["name"]
            )
        )

    @staticmethod
    def detect_tshark() -> Optional[str]:
        """Checks for installed TShark binary on the system."""
        for path in TSHARK_DEFAULT_PATHS:
            if os.path.exists(path):
                return path
        return None

    def _packet_callback(self, pkt) -> None:
        """Processes each captured packet into the FlowManager."""
        self.packet_count += 1
        try:
            # Check IP layer
            if IP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                proto = pkt[IP].proto
            elif IPv6 in pkt:
                src_ip = pkt[IPv6].src
                dst_ip = pkt[IPv6].dst
                proto = pkt[IPv6].nh
            else:
                return  # Non-IP packet

            length = len(pkt)
            timestamp = float(pkt.time)
            src_port = 0
            dst_port = 0
            header_len = 20
            tcp_flags = {}
            window_size = 0

            if TCP in pkt:
                src_port = int(pkt[TCP].sport)
                dst_port = int(pkt[TCP].dport)
                header_len = int(pkt[TCP].dataofs) * 4 if hasattr(pkt[TCP], "dataofs") else 20
                flags_int = int(pkt[TCP].flags)
                tcp_flags = {
                    "F": bool(flags_int & 0x01),
                    "S": bool(flags_int & 0x02),
                    "R": bool(flags_int & 0x04),
                    "P": bool(flags_int & 0x08),
                    "A": bool(flags_int & 0x10),
                    "U": bool(flags_int & 0x20),
                }
                window_size = int(pkt[TCP].window)
            elif UDP in pkt:
                src_port = int(pkt[UDP].sport)
                dst_port = int(pkt[UDP].dport)
                header_len = 8

            completed = self.flow_manager.process_packet(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto,
                length=length,
                timestamp=timestamp,
                header_len=header_len,
                tcp_flags=tcp_flags,
                window_size=window_size
            )
            if completed:
                self.completed_flows_queue.put(completed)

        except Exception as e:
            logger.debug(f"Error parsing packet: {e}")

    def start_live_capture(self, interface: Optional[str] = None, duration_seconds: float = 60.0) -> None:
        """Starts live packet capture for a fixed duration in a background thread."""
        if self.is_capturing:
            logger.warning("Capture is already in progress.")
            return
        if duration_seconds <= 0:
            raise ValueError("Capture duration must be greater than zero.")
        if self._capture_thread and self._capture_thread.is_alive():
            raise RuntimeError("The previous capture thread has not stopped yet.")

        self.is_capturing = True
        self.packet_count = 0
        self.capture_error = None

        def _worker():
            logger.info(f"Starting live capture on interface: {interface}...")
            deadline = time.monotonic() + duration_seconds
            try:
                while self.is_capturing:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    sniff(
                        iface=interface,
                        prn=self._packet_callback,
                        store=False,
                        count=0,
                        promisc=False,
                        timeout=min(self.CAPTURE_POLL_INTERVAL_SECONDS, remaining)
                    )
            except Exception as e:
                self.capture_error = str(e)
                logger.error(f"Live sniff failed: {e}")
            finally:
                self.is_capturing = False
                logger.info(f"Live capture ended. Total packets: {self.packet_count}")

        self._capture_thread = threading.Thread(target=_worker, daemon=True)
        self._capture_thread.start()

    def stop_live_capture(self) -> List[FlowRecord]:
        """Stops live capture and flushes all remaining flows."""
        self.is_capturing = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                raise RuntimeError("Live capture did not stop within two seconds.")
        return self.flow_manager.flush_all_flows()

    def read_pcap_file(
        self,
        pcap_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[FlowRecord]:
        """Reads and reconstructs network flows from a PCAP or PCAPNG file."""
        logger.info(f"Reading PCAP file: {pcap_path}")
        self.packet_count = 0
        reader = PcapReader(str(pcap_path))

        try:
            for pkt in reader:
                self._packet_callback(pkt)
                if progress_callback and (self.packet_count % 500 == 0):
                    progress_callback(self.packet_count)
        finally:
            reader.close()

        flows = self.flow_manager.flush_all_flows()
        logger.info(f"Reconstructed {len(flows)} flows from {self.packet_count} packets in {pcap_path.name}")
        return flows
