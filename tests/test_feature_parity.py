import unittest
from config import FEATURE_COLUMNS, FEATURE_COLUMNS_PATH
from network.flow_extractor import FlowRecord
from network.feature_extractor import FeatureExtractor
from utils.helpers import load_json

class TestFeatureParity(unittest.TestCase):
    """Verifies strict feature parity between flow extraction and model training schema."""

    def test_feature_count_and_order(self):
        """Checks that FlowRecord feature extraction produces exactly the 52 training features."""
        # Create a mock flow
        flow = FlowRecord(
            flow_id="192.168.1.5:443<->10.0.0.1:50000/6",
            src_ip="192.168.1.5",
            dst_ip="10.0.0.1",
            src_port=443,
            dst_port=50000,
            protocol=6,
            start_time=100.0
        )
        flow.add_packet(length=64, direction="fwd", timestamp=100.0, header_len=20)
        flow.add_packet(length=128, direction="bwd", timestamp=100.05, header_len=20)

        extracted_dict = FeatureExtractor.extract_features_from_flow(flow)
        extracted_keys = list(extracted_dict.keys())

        # Check total count
        self.assertEqual(len(extracted_keys), 52)

        # Check exact set match
        self.assertEqual(set(extracted_keys), set(FEATURE_COLUMNS))

        # Check DataFrame column order
        df = FeatureExtractor.flows_to_dataframe([flow])
        self.assertEqual(list(df.columns), FEATURE_COLUMNS)

    def test_schema_file_parity(self):
        """Checks that saved feature_columns.json exactly matches config.py."""
        saved_schema = load_json(FEATURE_COLUMNS_PATH)
        self.assertEqual(saved_schema["feature_count"], 52)
        self.assertEqual(saved_schema["features"], FEATURE_COLUMNS)

if __name__ == "__main__":
    unittest.main()
