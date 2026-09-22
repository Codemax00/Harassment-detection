"""
Unit tests for Detection and Multi-Object Tracking layers.
"""

import unittest
import numpy as np

from src.detection.detector import PersonDetection, YOLOPersonDetector, load_detector
from src.tracking.tracker import ByteTracker, TrackedPerson, compute_iou, KalmanBoxTracker


class TestDetectionAndTracking(unittest.TestCase):

    def test_person_detection_dataclass(self):
        det = PersonDetection(
            bounding_box=(100.0, 50.0, 200.0, 300.0),
            confidence=0.88,
            frame_id=1,
            track_candidate=True
        )
        self.assertEqual(det.x1, 100.0)
        self.assertEqual(det.y1, 50.0)
        self.assertEqual(det.x2, 200.0)
        self.assertEqual(det.y2, 300.0)
        self.assertEqual(det.width, 100.0)
        self.assertEqual(det.height, 250.0)
        self.assertEqual(det.center, (150.0, 175.0))
        self.assertEqual(det.area, 25000.0)

    def test_compute_iou(self):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0)

        box3 = (10.0, 10.0, 20.0, 20.0)
        self.assertAlmostEqual(compute_iou(box1, box3), 0.0)

        box4 = (5.0, 0.0, 15.0, 10.0)
        # Inter: 5x10 = 50. Union: 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
        self.assertAlmostEqual(compute_iou(box1, box4), 1.0 / 3.0)

    def test_kalman_box_tracker_prediction_update(self):
        tracker = KalmanBoxTracker((50.0, 50.0, 150.0, 200.0))
        pred_box = tracker.predict()
        self.assertEqual(len(pred_box), 4)

        # Update with slightly moved box
        tracker.update((55.0, 50.0, 155.0, 200.0))
        updated_box = tracker.get_bbox()
        self.assertTrue(updated_box[0] > 40.0)

    def test_bytetrack_persistent_id(self):
        tracker = ByteTracker(min_hits=1)

        # Frame 1: Person at (100, 100, 150, 250)
        dets_f1 = [
            PersonDetection(bounding_box=(100.0, 100.0, 150.0, 250.0), confidence=0.9, frame_id=1)
        ]
        tracks_f1 = tracker.update(dets_f1, timestamp=0.033, frame_id=1)
        self.assertEqual(len(tracks_f1), 1)
        initial_id = tracks_f1[0].track_id

        # Frame 2: Person moved slightly to (105, 100, 155, 250)
        dets_f2 = [
            PersonDetection(bounding_box=(105.0, 100.0, 155.0, 250.0), confidence=0.88, frame_id=2)
        ]
        tracks_f2 = tracker.update(dets_f2, timestamp=0.066, frame_id=2)
        self.assertEqual(len(tracks_f2), 1)
        # ID must persist across consecutive frames
        self.assertEqual(tracks_f2[0].track_id, initial_id)


if __name__ == "__main__":
    unittest.main()
