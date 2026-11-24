import time
from typing import Dict, Tuple, List, Optional
from config import FLOW_TIMEOUT_SECONDS
from network.flow_extractor import FlowRecord

class FlowManager:
    """Tracks active network flows, detects timeouts, and terminates flows on FIN/RST."""

    def __init__(self, timeout: float = FLOW_TIMEOUT_SECONDS):
        self.timeout = timeout
        self.active_flows: Dict[Tuple, FlowRecord] = {}

    @staticmethod
    def _normalize_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: int) -> Tuple[Tuple, str]:
        """
        Creates a bidirectional key so forward and reverse packets map to the same flow.
        Returns: (flow_key, direction where 'fwd' is original direction and 'bwd' is reverse).
        """
        endpoint_a = (src_ip, src_port)
        endpoint_b = (dst_ip, dst_port)
        if endpoint_a <= endpoint_b:
            return (endpoint_a, endpoint_b, proto), "fwd"
        else:
            return (endpoint_b, endpoint_a, proto), "bwd"

    def process_packet(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: int,
        length: int,
        timestamp: float,
        header_len: int = 20,
        tcp_flags: Optional[Dict[str, bool]] = None,
        window_size: int = 0
    ) -> Optional[FlowRecord]:
        """
        Processes a single packet into its flow.
        If flow terminates (FIN/RST), returns the completed FlowRecord.
        """
        key, _ = self._normalize_key(src_ip, src_port, dst_ip, dst_port, protocol)

        if key not in self.active_flows:
            flow_id = f"{src_ip}:{src_port}<->{dst_ip}:{dst_port}/{protocol}"
            flow = FlowRecord(
                flow_id=flow_id,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                start_time=timestamp
            )
            self.active_flows[key] = flow
            # When creating a new flow, the current packet defines forward direction
            actual_direction = "fwd"
        else:
            flow = self.active_flows[key]
            # Check if packet matches flow forward direction
            actual_direction = "fwd" if (src_ip == flow.src_ip and src_port == flow.src_port) else "bwd"

        flow.add_packet(
            length=length,
            direction=actual_direction,
            timestamp=timestamp,
            header_len=header_len,
            tcp_flags=tcp_flags,
            window_size=window_size
        )

        # Check for immediate TCP termination
        if tcp_flags and (tcp_flags.get("F", False) or tcp_flags.get("R", False)):
            # If flow has accumulated both forward and backward packets or sufficient length, terminate
            if flow.fwd_packets > 1 or flow.bwd_packets > 1:
                return self.active_flows.pop(key, None)

        return None

    def get_expired_flows(self, current_time: Optional[float] = None) -> List[FlowRecord]:
        """Finds and removes flows exceeding the inactivity timeout."""
        now = current_time if current_time is not None else time.time()
        expired_keys = [
            k for k, flow in self.active_flows.items() if (now - flow.last_time) > self.timeout
        ]
        expired_flows = [self.active_flows.pop(k) for k in expired_keys]
        return expired_flows

    def flush_all_flows(self) -> List[FlowRecord]:
        """Flushes and returns all remaining active flows."""
        flows = list(self.active_flows.values())
        self.active_flows.clear()
        return flows
