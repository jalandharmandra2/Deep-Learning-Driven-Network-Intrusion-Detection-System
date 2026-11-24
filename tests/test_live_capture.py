import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scapy.all import IP, TCP

from network.packet_capture import PacketCapture


class TestLiveCapture(unittest.TestCase):
    def test_capture_stops_after_duration_without_receiving_packets(self):
        capture = PacketCapture()

        def wait_for_timeout(**kwargs):
            time.sleep(kwargs["timeout"])

        with patch("network.packet_capture.sniff", side_effect=wait_for_timeout) as sniff:
            capture.start_live_capture(interface="test-interface", duration_seconds=0.05)
            capture._capture_thread.join(timeout=1)

        self.assertFalse(capture.is_capturing)
        self.assertEqual(capture.packet_count, 0)
        self.assertIsNone(capture.capture_error)
        self.assertEqual(sniff.call_args.kwargs["count"], 0)
        self.assertFalse(sniff.call_args.kwargs["promisc"])

    def test_capture_error_is_available_to_the_ui(self):
        capture = PacketCapture()

        with patch("network.packet_capture.sniff", side_effect=OSError("Npcap unavailable")):
            capture.start_live_capture(interface="test-interface", duration_seconds=1)
            capture._capture_thread.join(timeout=1)

        self.assertFalse(capture.is_capturing)
        self.assertEqual(capture.capture_error, "Npcap unavailable")

    def test_captured_packet_is_counted_and_added_to_a_flow(self):
        capture = PacketCapture()
        emitted_packet = False

        def emit_packet(**kwargs):
            nonlocal emitted_packet
            if not emitted_packet:
                packet = (
                    IP(src="192.168.0.10", dst="192.168.0.20")
                    / TCP(sport=1234, dport=80, dataofs=5)
                )
                kwargs["prn"](packet)
                emitted_packet = True
            time.sleep(kwargs["timeout"])

        with patch("network.packet_capture.sniff", side_effect=emit_packet):
            capture.start_live_capture(interface="test-interface", duration_seconds=0.05)
            capture._capture_thread.join(timeout=1)

        self.assertEqual(capture.packet_count, 1)
        self.assertEqual(len(capture.flow_manager.active_flows), 1)

    def test_active_ip_interface_is_listed_before_unconfigured_adapters(self):
        capture_interfaces = {
            "wan-monitor": SimpleNamespace(description="WAN Miniport", ip=""),
            "loopback": SimpleNamespace(description="Loopback", ip="127.0.0.1"),
            "wifi": SimpleNamespace(description="Wi-Fi", ip="192.168.0.102"),
            "link-local": SimpleNamespace(description="Virtual adapter", ip="169.254.1.1"),
        }

        with patch("network.packet_capture.conf.ifaces", capture_interfaces):
            interfaces = PacketCapture.list_interfaces()

        self.assertEqual(interfaces[0]["name"], "wifi")
        self.assertEqual(interfaces[-1]["name"], "wan-monitor")

    def test_capture_duration_must_be_positive(self):
        capture = PacketCapture()

        with self.assertRaises(ValueError):
            capture.start_live_capture(duration_seconds=0)


if __name__ == "__main__":
    unittest.main()
