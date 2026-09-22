"""
End-to-End Pipeline Integration Test.
"""

import unittest
import numpy as np

from src.pipeline import GuardianMatrixPipeline, PipelineFrameResult
from src.policy.safety_policy import PolicyState


class TestEndToEndPipeline(unittest.TestCase):

    def test_pipeline_frame_processing(self):
        pipeline = GuardianMatrixPipeline(video_id="TEST_STREAM")

        # Create synthetic test frame (640x480 RGB image)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result: PipelineFrameResult = pipeline.process_frame(
            frame=dummy_frame,
            timestamp=0.033,
            frame_id=1,
            annotate=True
        )

        self.assertEqual(result.frame_id, 1)
        self.assertAlmostEqual(result.timestamp, 0.033)
        self.assertIsNotNone(result.evidence)
        self.assertIsNotNone(result.system1_decision)
        self.assertIsNotNone(result.policy_action)
        self.assertIsNotNone(result.annotated_frame)
        self.assertEqual(result.annotated_frame.shape, (480, 640, 3))
        self.assertTrue(result.processing_time_ms > 0.0)


if __name__ == "__main__":
    unittest.main()
