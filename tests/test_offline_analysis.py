import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from dashboard.offline_analysis import _classify_uploaded_file


class TestOfflineAnalysis(unittest.TestCase):
    def test_csv_upload_uses_flow_classifier(self):
        uploaded_file = SimpleNamespace(name="flows.csv")
        input_data = pd.DataFrame({"feature": [1.0]})
        classifier = Mock()
        classifier.classify_dataframe.return_value = [{"predicted_class": "BENIGN"}]

        with patch(
            "dashboard.offline_analysis.pd.read_csv",
            return_value=input_data
        ) as read_csv:
            results, returned_data = _classify_uploaded_file(uploaded_file, classifier)

        read_csv.assert_called_once_with(
            uploaded_file,
            nrows=10000,
            encoding="latin-1"
        )
        classifier.classify_dataframe.assert_called_once_with(input_data)
        self.assertEqual(results, [{"predicted_class": "BENIGN"}])
        self.assertIs(returned_data, input_data)

    def test_pcapng_upload_reconstructs_flows_and_removes_temporary_file(self):
        uploaded_file = SimpleNamespace(
            name="capture.PCAPNG",
            getbuffer=Mock(return_value=b"pcap data")
        )
        capture = Mock()
        capture.read_pcap_file.return_value = ["flow"]
        classifier = Mock()
        classifier.classify_flows.return_value = [{"predicted_class": "BENIGN"}]

        with patch(
            "dashboard.offline_analysis.PacketCapture",
            return_value=capture
        ):
            results, input_data = _classify_uploaded_file(uploaded_file, classifier)

        pcap_path = capture.read_pcap_file.call_args.args[0]
        self.assertEqual(Path(pcap_path).suffix, ".pcapng")
        self.assertFalse(Path(pcap_path).exists())
        classifier.classify_flows.assert_called_once_with(["flow"])
        self.assertEqual(results, [{"predicted_class": "BENIGN"}])
        self.assertIsNone(input_data)


if __name__ == "__main__":
    unittest.main()
