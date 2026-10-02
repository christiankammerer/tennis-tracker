import unittest
from pathlib import Path

import numpy as np

from detect.court import _touches_broadcast_overlay, detect_court

FRAME_SHAPE = (720, 1280, 3)

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

# detect_court(tennis2.mp4 frame 1500) before the overlay filter.
# near-left sits on the DJOKOVIC score graphic (y=620); H is sheared.
FRAME_1500_SCOREBUG = np.array(
    [
        [181.3, 620.2],
        [1049.5, 555.1],
        [889.0, 251.8],
        [370.3, 266.9],
    ],
    dtype=np.float32,
)

# detect_court(tennis2.mp4 frame 1800) before the overlay filter.
# Three of four corners lie on the score graphic; the quad collapses.
FRAME_1800_SCOREBUG = np.array(
    [
        [292.5, 627.3],
        [1075.2, 647.3],
        [933.0, 377.1],
        [292.9, 626.4],
    ],
    dtype=np.float32,
)

VIDEO = Path(__file__).resolve().parents[1] / "tennis2.mp4"


class TestBroadcastOverlayFilter(unittest.TestCase):
    def test_sample_clicks_are_above_the_scorebug(self):
        self.assertFalse(_touches_broadcast_overlay(IMAGE_CORNERS, FRAME_SHAPE))

    def test_rejects_frame_1500_corner_on_score_graphic(self):
        self.assertTrue(_touches_broadcast_overlay(FRAME_1500_SCOREBUG, FRAME_SHAPE))

    def test_rejects_frame_1800_collapsed_overlay_quad(self):
        self.assertTrue(_touches_broadcast_overlay(FRAME_1800_SCOREBUG, FRAME_SHAPE))

    def test_detect_court_frame_1500_does_not_anchor_on_scorebug(self):
        import cv2

        if not VIDEO.exists():
            self.skipTest(f"missing {VIDEO}")
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.skipTest("could not read frame 1500")
        det = detect_court(frame)
        self.assertFalse(_touches_broadcast_overlay(det.corners, frame.shape), det.corners)
        self.assertTrue(np.all(det.corners[:, 1] <= 0.85 * frame.shape[0]))

    def test_detect_court_frame_1800_recovers_the_court(self):
        import cv2

        if not VIDEO.exists():
            self.skipTest(f"missing {VIDEO}")
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1800)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.skipTest("could not read frame 1800")
        det = detect_court(frame)
        self.assertFalse(_touches_broadcast_overlay(det.corners, frame.shape), det.corners)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        near_y = 0.5 * (float(det.corners[0, 1]) + float(det.corners[1, 1]))
        # Before the filter this returned a triangle on the scorebug (near_y≈637).
        self.assertLess(far_y, 200.0, det.corners)
        self.assertGreater(near_y, 500.0, det.corners)


if __name__ == "__main__":
    unittest.main()
