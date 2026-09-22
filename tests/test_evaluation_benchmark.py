"""
Unit tests for Evaluation Suite and Ablation Studies.
"""

import unittest

from src.evaluation.benchmark import EvaluationSuite, AblationStudyRunner


class TestEvaluationBenchmark(unittest.TestCase):

    def test_classification_metrics(self):
        y_true = ["NORMAL_ACTIVITY", "NORMAL_ACTIVITY", "REPEATED_CONTACT_PATTERN", "PURSUIT_PATTERN"]
        y_pred = ["NORMAL_ACTIVITY", "NORMAL_ACTIVITY", "REPEATED_CONTACT_PATTERN", "NORMAL_ACTIVITY"]

        metrics = EvaluationSuite.compute_classification_metrics(
            y_true, y_pred, classes=["NORMAL_ACTIVITY", "REPEATED_CONTACT_PATTERN", "PURSUIT_PATTERN"]
        )

        self.assertAlmostEqual(metrics.accuracy, 0.75, places=2)
        self.assertIn("NORMAL_ACTIVITY", metrics.confusion_matrix)
        self.assertIn("REPEATED_CONTACT_PATTERN", metrics.per_class_f1)

    def test_safety_metrics_false_alarms_per_hour(self):
        # 1 hour duration (3600s), 4 alerts predicted, 1 matched ground truth, 3 false alerts
        duration_s = 3600.0
        predicted_alerts = [
            (100.0, "ALERT"),
            (500.0, "ALERT"),
            (1200.0, "ALERT"),
            (2500.0, "ALERT")
        ]
        ground_truth = [
            (95.0, 105.0, "ALERT")  # Matches 100.0
        ]

        metrics = EvaluationSuite.compute_safety_metrics(duration_s, predicted_alerts, ground_truth)
        # 3 false alerts in 1 hour = 3.0 false alarms/hour
        self.assertAlmostEqual(metrics.false_alarms_per_hour, 3.0, places=1)
        self.assertEqual(metrics.total_alerts, 4)
        self.assertEqual(metrics.false_alerts, 3)

    def test_ablation_study_runner(self):
        runner = AblationStudyRunner()
        ablations = runner.run_all_ablations()
        self.assertEqual(len(ablations), 6)
        # Verify that upgraded architecture achieves higher F1 and lower false alarms than baseline
        baseline = ablations[0]
        upgraded = ablations[-1]
        self.assertTrue(upgraded.f1_score > baseline.f1_score)
        self.assertTrue(upgraded.false_alarms_per_hour < baseline.false_alarms_per_hour)


if __name__ == "__main__":
    unittest.main()
