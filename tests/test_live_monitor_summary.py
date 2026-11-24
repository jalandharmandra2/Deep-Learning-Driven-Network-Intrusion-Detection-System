import unittest

from dashboard.threat_summary import summarize_detections


class TestLiveMonitorSummary(unittest.TestCase):
    def test_summarizes_malicious_flows_by_attack_type(self):
        malicious, attack_counts = summarize_detections([
            {"status": "BENIGN", "predicted_class": "BENIGN"},
            {"status": "MALICIOUS", "predicted_class": "FLOODING"},
            {"status": "MALICIOUS", "predicted_class": "FLOODING"},
            {"status": "MALICIOUS", "predicted_class": "SCANNING"},
        ])

        self.assertEqual(len(malicious), 3)
        self.assertEqual(attack_counts, {"FLOODING": 2, "SCANNING": 1})

    def test_empty_and_benign_results_have_no_malicious_detections(self):
        for results in ([], [{"status": "BENIGN", "predicted_class": "BENIGN"}]):
            with self.subTest(results=results):
                malicious, attack_counts = summarize_detections(results)
                self.assertEqual(malicious, [])
                self.assertEqual(attack_counts, {})


if __name__ == "__main__":
    unittest.main()
