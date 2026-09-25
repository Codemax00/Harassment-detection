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

    def test_appearance_feature_extraction_and_similarity(self):
        from src.tracking.tracker import extract_appearance_feature, cosine_similarity

        # Create two distinct test images:
        # Person A: Red upper body, Blue lower body
        img_a = np.zeros((200, 200, 3), dtype=np.uint8)
        img_a[0:100, :, 2] = 220    # Red in BGR
        img_a[100:200, :, 0] = 220  # Blue in BGR

        # Person B: Green upper body, Yellow lower body
        img_b = np.zeros((200, 200, 3), dtype=np.uint8)
        img_b[0:100, :, 1] = 220    # Green in BGR
        img_b[100:200, :, 0] = 200  # Yellow (B=0, G=200, R=200)
        img_b[100:200, :, 1] = 200
        img_b[100:200, :, 2] = 200

        box = (20.0, 20.0, 180.0, 180.0)
        feat_a1 = extract_appearance_feature(img_a, box)
        feat_a2 = extract_appearance_feature(img_a, (25.0, 25.0, 175.0, 175.0))
        feat_b = extract_appearance_feature(img_b, box)

        self.assertIsNotNone(feat_a1)
        self.assertIsNotNone(feat_a2)
        self.assertIsNotNone(feat_b)

        # Same person with slight bbox shift should have very high similarity (> 0.90)
        sim_same = cosine_similarity(feat_a1, feat_a2)
        self.assertGreater(sim_same, 0.90)

        # Distinct people with different clothes should have significantly lower similarity
        sim_diff = cosine_similarity(feat_a1, feat_b)
        self.assertLess(sim_diff, sim_same)

    def test_lost_track_recovery_with_visual_reid(self):
        """
        Verify that when a person is temporarily occluded / missing for multiple frames,
        re-detection with matching appearance recovers the original track ID instead
        of spawning a new ID (preventing ID explosion).
        """
        tracker = ByteTracker(min_hits=1, max_time_lost=30)

        # Create a test frame with a distinct colored person
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Person patch at (100, 100, 160, 260): Red jacket, Blue jeans
        frame[100:180, 100:160, 2] = 240
        frame[180:260, 100:160, 0] = 240

        det_f1 = [PersonDetection(bounding_box=(100.0, 100.0, 160.0, 260.0), confidence=0.92, frame_id=1)]
        tracks_f1 = tracker.update(det_f1, timestamp=0.033, frame_id=1, frame=frame)
        self.assertEqual(len(tracks_f1), 1)
        initial_id = tracks_f1[0].track_id

        # Frames 2-5: Person is completely occluded (0 detections)
        for f in range(2, 6):
            empty_tracks = tracker.update([], timestamp=f * 0.033, frame_id=f, frame=frame)
            self.assertEqual(len(empty_tracks), 0)

        # Frame 6: Person re-emerges nearby at (120, 105, 180, 265)
        # Move the colored patch to match the new position
        frame_f6 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_f6[105:185, 120:180, 2] = 240
        frame_f6[185:265, 120:180, 0] = 240

        det_f6 = [PersonDetection(bounding_box=(120.0, 105.0, 180.0, 265.0), confidence=0.90, frame_id=6)]
        tracks_f6 = tracker.update(det_f6, timestamp=0.200, frame_id=6, frame=frame_f6)

        # Track should be recovered with the EXACT original track ID!
        self.assertEqual(len(tracks_f6), 1)
        self.assertEqual(tracks_f6[0].track_id, initial_id, "Track ID must be recovered via Re-ID, not incremented!")


if __name__ == "__main__":
    unittest.main()

