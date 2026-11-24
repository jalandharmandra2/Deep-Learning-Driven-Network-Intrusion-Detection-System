from typing import Dict, List
import pandas as pd
from config import FEATURE_COLUMNS
from network.flow_extractor import FlowRecord

class FeatureExtractor:
    """Extracts the exact 52 model features from FlowRecord objects."""

    @staticmethod
    def extract_features_from_flow(flow: FlowRecord) -> Dict[str, float]:
        """Calculates and maps the 52 features from a single FlowRecord."""
        duration_us = max(1.0, (flow.last_time - flow.start_time) * 1_000_000.0)
        duration_s = duration_us / 1_000_000.0

        all_lengths = flow.fwd_lengths + flow.bwd_lengths
        tot_fwd_pkts = flow.fwd_packets
        tot_bwd_pkts = flow.bwd_packets
        tot_pkts = tot_fwd_pkts + tot_bwd_pkts

        tot_fwd_len = sum(flow.fwd_lengths)
        tot_bwd_len = sum(flow.bwd_lengths)
        tot_bytes = tot_fwd_len + tot_bwd_len

        fwd_len_stats = FlowRecord._stats(flow.fwd_lengths)
        bwd_len_stats = FlowRecord._stats(flow.bwd_lengths)
        pkt_len_stats = FlowRecord._stats(all_lengths)

        flow_iat_stats = FlowRecord._stats(flow.flow_iats)
        fwd_iat_stats = FlowRecord._stats(flow.fwd_iats)
        bwd_iat_stats = FlowRecord._stats(flow.bwd_iats)

        active_stats = FlowRecord._stats(flow.active_times)
        idle_stats = FlowRecord._stats(flow.idle_times)

        # Flow rates
        flow_bytes_per_s = (tot_bytes / duration_s) if duration_s > 0 else 0.0
        flow_pkts_per_s = (tot_pkts / duration_s) if duration_s > 0 else 0.0
        fwd_pkts_per_s = (tot_fwd_pkts / duration_s) if duration_s > 0 else 0.0
        bwd_pkts_per_s = (tot_bwd_pkts / duration_s) if duration_s > 0 else 0.0
        down_up_ratio = (tot_bwd_pkts / tot_fwd_pkts) if tot_fwd_pkts > 0 else 0.0

        features = {
            "Destination Port": float(flow.dst_port),
            "Flow Duration": float(duration_us),
            "Total Fwd Packets": float(tot_fwd_pkts),
            "Total Backward Packets": float(tot_bwd_pkts),
            "Total Length of Fwd Packets": float(tot_fwd_len),
            "Total Length of Bwd Packets": float(tot_bwd_len),
            "Fwd Packet Length Max": float(fwd_len_stats["max"]),
            "Fwd Packet Length Min": float(fwd_len_stats["min"]),
            "Fwd Packet Length Mean": float(fwd_len_stats["mean"]),
            "Fwd Packet Length Std": float(fwd_len_stats["std"]),
            "Bwd Packet Length Max": float(bwd_len_stats["max"]),
            "Bwd Packet Length Min": float(bwd_len_stats["min"]),
            "Bwd Packet Length Mean": float(bwd_len_stats["mean"]),
            "Bwd Packet Length Std": float(bwd_len_stats["std"]),
            "Flow Bytes/s": float(flow_bytes_per_s),
            "Flow Packets/s": float(flow_pkts_per_s),
            "Flow IAT Mean": float(flow_iat_stats["mean"]),
            "Flow IAT Std": float(flow_iat_stats["std"]),
            "Flow IAT Max": float(flow_iat_stats["max"]),
            "Flow IAT Min": float(flow_iat_stats["min"]),
            "Fwd IAT Total": float(sum(flow.fwd_iats)),
            "Fwd IAT Mean": float(fwd_iat_stats["mean"]),
            "Fwd IAT Std": float(fwd_iat_stats["std"]),
            "Fwd IAT Max": float(fwd_iat_stats["max"]),
            "Fwd IAT Min": float(fwd_iat_stats["min"]),
            "Bwd IAT Total": float(sum(flow.bwd_iats)),
            "Bwd IAT Mean": float(bwd_iat_stats["mean"]),
            "Bwd IAT Std": float(bwd_iat_stats["std"]),
            "Bwd IAT Max": float(bwd_iat_stats["max"]),
            "Bwd IAT Min": float(bwd_iat_stats["min"]),
            "Fwd Header Length": float(flow.fwd_header_len),
            "Bwd Header Length": float(flow.bwd_header_len),
            "Fwd Packets/s": float(fwd_pkts_per_s),
            "Bwd Packets/s": float(bwd_pkts_per_s),
            "Min Packet Length": float(pkt_len_stats["min"]),
            "Max Packet Length": float(pkt_len_stats["max"]),
            "Packet Length Mean": float(pkt_len_stats["mean"]),
            "Packet Length Std": float(pkt_len_stats["std"]),
            "FIN Flag Count": float(flow.fin_count),
            "SYN Flag Count": float(flow.syn_count),
            "RST Flag Count": float(flow.rst_count),
            "PSH Flag Count": float(flow.psh_count),
            "ACK Flag Count": float(flow.ack_count),
            "URG Flag Count": float(flow.urg_count),
            "Down/Up Ratio": float(down_up_ratio),
            "Init_Win_bytes_forward": float(flow.init_win_bytes_fwd),
            "Init_Win_bytes_backward": float(flow.init_win_bytes_bwd),
            "act_data_pkt_fwd": float(flow.act_data_pkt_fwd),
            "Active Mean": float(active_stats["mean"]),
            "Active Max": float(active_stats["max"]),
            "Idle Mean": float(idle_stats["mean"]),
            "Idle Max": float(idle_stats["max"])
        }
        return features

    @classmethod
    def flows_to_dataframe(cls, flows: List[FlowRecord]) -> pd.DataFrame:
        """Converts a list of FlowRecord objects to a pandas DataFrame strictly matching FEATURE_COLUMNS."""
        if not flows:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        records = [cls.extract_features_from_flow(f) for f in flows]
        df = pd.DataFrame(records)
        return df[FEATURE_COLUMNS]
