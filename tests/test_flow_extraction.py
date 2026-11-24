import unittest
from network.flow_manager import FlowManager
from network.feature_extractor import FeatureExtractor

class TestFlowExtraction(unittest.TestCase):
    """Tests packet parsing, bidirectional flow tracking, and feature calculation."""

    def setUp(self):
        self.flow_mgr = FlowManager(timeout=5.0)

    def test_bidirectional_flow_aggregation(self):
        """Tests that forward and backward packets are correctly grouped into a single flow."""
        # Forward packet: 192.168.1.10:1234 -> 93.184.216.34:80
        self.flow_mgr.process_packet(
            src_ip="192.168.1.10",
            dst_ip="93.184.216.34",
            src_port=1234,
            dst_port=80,
            protocol=6,
            length=60,
            timestamp=100.0,
            tcp_flags={"S": True}
        )

        # Backward packet (reply): 93.184.216.34:80 -> 192.168.1.10:1234
        self.flow_mgr.process_packet(
            src_ip="93.184.216.34",
            dst_ip="192.168.1.10",
            src_port=80,
            dst_port=1234,
            protocol=6,
            length=1500,
            timestamp=100.02,
            tcp_flags={"S": True, "A": True}
        )

        flows = self.flow_mgr.flush_all_flows()
        self.assertEqual(len(flows), 1)

        flow = flows[0]
        self.assertEqual(flow.fwd_packets, 1)
        self.assertEqual(flow.bwd_packets, 1)
        self.assertEqual(flow.syn_count, 2)
        self.assertEqual(flow.ack_count, 1)

        # Verify feature extraction
        feats = FeatureExtractor.extract_features_from_flow(flow)
        self.assertEqual(feats["Total Fwd Packets"], 1.0)
        self.assertEqual(feats["Total Backward Packets"], 1.0)
        self.assertEqual(feats["Total Length of Fwd Packets"], 60.0)
        self.assertEqual(feats["Total Length of Bwd Packets"], 1500.0)
        self.assertEqual(len(feats), 52)

if __name__ == "__main__":
    unittest.main()
