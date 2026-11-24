import math
from typing import List, Dict, Optional

class FlowRecord:
    """Maintains packet-level statistics for a bidirectional network flow."""

    def __init__(
        self,
        flow_id: str,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: int,
        start_time: float
    ):
        self.flow_id = flow_id
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.start_time = start_time
        self.last_time = start_time

        # Packets and lengths
        self.fwd_packets: int = 0
        self.bwd_packets: int = 0
        self.fwd_lengths: List[int] = []
        self.bwd_lengths: List[int] = []

        # Inter-arrival times (in microseconds)
        self.flow_iats: List[float] = []
        self.fwd_iats: List[float] = []
        self.bwd_iats: List[float] = []
        self.last_fwd_time: Optional[float] = None
        self.last_bwd_time: Optional[float] = None

        # Headers and Windows
        self.fwd_header_len: int = 0
        self.bwd_header_len: int = 0
        self.init_win_bytes_fwd: int = 0
        self.init_win_bytes_bwd: int = 0
        self.act_data_pkt_fwd: int = 0

        # TCP Flags
        self.fin_count: int = 0
        self.syn_count: int = 0
        self.rst_count: int = 0
        self.psh_count: int = 0
        self.ack_count: int = 0
        self.urg_count: int = 0

        # Activity and Idleness (threshold = 1.0s)
        self.active_times: List[float] = []
        self.idle_times: List[float] = []
        self.last_active_start: float = start_time

    def add_packet(
        self,
        length: int,
        direction: str,  # 'fwd' or 'bwd'
        timestamp: float,
        header_len: int = 20,
        tcp_flags: Optional[Dict[str, bool]] = None,
        window_size: int = 0
    ) -> None:
        """Updates flow state with a new packet."""
        # Calculate Flow IAT in microseconds
        if self.last_time > 0 and timestamp >= self.last_time:
            iat_us = (timestamp - self.last_time) * 1_000_000.0
            self.flow_iats.append(iat_us)

            # Check idle/active transition
            time_gap_s = timestamp - self.last_time
            if time_gap_s > 1.0:
                self.idle_times.append(time_gap_s * 1_000_000.0)
                active_s = self.last_time - self.last_active_start
                if active_s > 0:
                    self.active_times.append(active_s * 1_000_000.0)
                self.last_active_start = timestamp

        self.last_time = max(self.last_time, timestamp)

        if direction == "fwd":
            self.fwd_packets += 1
            self.fwd_lengths.append(length)
            self.fwd_header_len += header_len
            if self.init_win_bytes_fwd == 0 and window_size > 0:
                self.init_win_bytes_fwd = window_size
            if length > header_len:
                self.act_data_pkt_fwd += 1

            if self.last_fwd_time is not None and timestamp >= self.last_fwd_time:
                self.fwd_iats.append((timestamp - self.last_fwd_time) * 1_000_000.0)
            self.last_fwd_time = timestamp

        else:  # bwd
            self.bwd_packets += 1
            self.bwd_lengths.append(length)
            self.bwd_header_len += header_len
            if self.init_win_bytes_bwd == 0 and window_size > 0:
                self.init_win_bytes_bwd = window_size

            if self.last_bwd_time is not None and timestamp >= self.last_bwd_time:
                self.bwd_iats.append((timestamp - self.last_bwd_time) * 1_000_000.0)
            self.last_bwd_time = timestamp

        # TCP Flags
        if tcp_flags:
            if tcp_flags.get("F", False): self.fin_count += 1
            if tcp_flags.get("S", False): self.syn_count += 1
            if tcp_flags.get("R", False): self.rst_count += 1
            if tcp_flags.get("P", False): self.psh_count += 1
            if tcp_flags.get("A", False): self.ack_count += 1
            if tcp_flags.get("U", False): self.urg_count += 1

    @staticmethod
    def _stats(arr: List[float]) -> Dict[str, float]:
        """Calculates min, max, mean, std for a list of values."""
        if not arr:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        n = len(arr)
        min_v = float(min(arr))
        max_v = float(max(arr))
        mean_v = float(sum(arr)) / n
        if n > 1:
            var_v = sum((x - mean_v) ** 2 for x in arr) / (n - 1)
            std_v = math.sqrt(max(0.0, var_v))
        else:
            std_v = 0.0
        return {"min": min_v, "max": max_v, "mean": mean_v, "std": std_v}
