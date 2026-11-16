import unittest
from pathlib import Path

import numpy as np

from detect.court import _far_edge_past_baseline, detect_court

# Pixel corners from the sample frame (near-left, near-right, far-right, far-left).
IMAGE_CORNERS = np.array(
    [
        [279.6, 502.8],
        [1002.4, 511.8],
        [824.7, 167.8],
        [458.3, 164.8],
    ],
    dtype=np.float32,
)

# detect_court(tennis2.mp4 frame 1050) before the aspect cap.
# Far edge is the back surround (y=146), not the far baseline (y≈189).
# Homography maps a bounce on the true far baseline to court y ≈ 5.5 m.
FRAME_1050_OVERSHOOT = np.array(
    [
        [289.28265, 510.96597],
        [986.33765, 521.50696],
        [789.9592, 146.0],
        [483.9631, 146.0],
    ],
    dtype=np.float32,
)

# Same frame, same sidelines, but paired with the true far baseline.
FRAME_1050_FULL = np.array(
    [
        [289.28265, 510.96597],
        [986.33765, 521.50696],
        [812.62085, 189.33267],
        [460.92847, 189.18289],
    ],
    dtype=np.float32,
)

VIDEO = Path(__file__).resolve().parents[1] / "tennis2.mp4"
FRAME_0 = Path(__file__).resolve().parents[1] / "debug" / "frame_0.jpg"


class TestFarEdgePastBaseline(unittest.TestCase):
    def test_sample_clicks_are_not_past_the_far_baseline(self):
        self.assertFalse(_far_edge_past_baseline(IMAGE_CORNERS))

    def test_rejects_back_surround_as_far_baseline(self):
        self.assertTrue(_far_edge_past_baseline(FRAME_1050_OVERSHOOT))

    def test_accepts_true_far_baseline_on_same_frame(self):
        self.assertFalse(_far_edge_past_baseline(FRAME_1050_FULL))

    def test_detect_court_frame_1050_uses_far_baseline(self):
        import cv2

        if not VIDEO.exists():
            self.skipTest(f"missing {VIDEO}")
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1050)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.skipTest("could not read frame 1050")
        det = detect_court(frame)
        self.assertFalse(_far_edge_past_baseline(det.corners), det.corners)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        near_y = 0.5 * (float(det.corners[0, 1]) + float(det.corners[1, 1]))
        # Before the filter this returned far_y=146 (back surround).
        self.assertGreater(far_y, 170.0, det.corners)
        self.assertLess(far_y, 220.0, det.corners)
        self.assertGreater(near_y, 490.0, det.corners)
        np.testing.assert_allclose(det.corners, FRAME_1050_FULL, atol=3.0)

    def test_detect_court_on_sample_frame_keeps_far_baseline(self):
        import cv2

        if not FRAME_0.exists():
            self.skipTest(f"missing {FRAME_0}")
        frame = cv2.imread(str(FRAME_0))
        det = detect_court(frame)
        self.assertFalse(_far_edge_past_baseline(det.corners), det.corners)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        self.assertGreater(float(far_y), 150.0)
        self.assertLess(float(far_y), 200.0)
        np.testing.assert_allclose(det.corners, IMAGE_CORNERS, atol=5.0)

    def test_detect_court_frame_300_still_full_court(self):
        import cv2

        if not VIDEO.exists():
            self.skipTest(f"missing {VIDEO}")
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.skipTest("could not read frame 300")
        det = detect_court(frame)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        near_y = 0.5 * (float(det.corners[0, 1]) + float(det.corners[1, 1]))
        self.assertGreater(far_y, 140.0, det.corners)
        self.assertLess(far_y, 200.0, det.corners)
        self.assertGreater(near_y, 520.0, det.corners)
        self.assertFalse(_far_edge_past_baseline(det.corners), det.corners)


if __name__ == "__main__":
    unittest.main()
