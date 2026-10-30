import unittest
from pathlib import Path

import cv2
import numpy as np

from detect.court import (
    MAX_QUAD_INTERIOR_WHITE,
    _quad_interior_white_fraction,
    detect_court,
)

FRAME_0 = Path(__file__).resolve().parents[1] / "debug" / "frame_0.jpg"
VIDEO = Path(__file__).resolve().parents[1] / "tennis.mp4"

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


def _bumper_frame(h: int = 720, w: int = 1280) -> np.ndarray:
    """White broadcast card with dark geometric borders (US Open bumper)."""
    frame = np.full((h, w, 3), 255, dtype=np.uint8)
    blue = (180, 80, 20)
    cv2.fillConvexPoly(frame, np.array([[0, 0], [420, 0], [0, 280]], np.int32), blue)
    cv2.fillConvexPoly(frame, np.array([[w, 0], [w - 420, 0], [w, 280]], np.int32), blue)
    cv2.fillConvexPoly(frame, np.array([[0, h], [420, h], [0, h - 280]], np.int32), blue)
    cv2.fillConvexPoly(frame, np.array([[w, h], [w - 420, h], [w, h - 280]], np.int32), blue)
    cv2.putText(frame, "#USOpen", (440, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.4, blue, 8)
    return frame


class TestWhiteBroadcastGraphic(unittest.TestCase):
    def test_interior_fraction_is_low_on_painted_lines_only(self):
        white = np.zeros((200, 200), dtype=np.uint8)
        white[20:22, 20:180] = 255
        white[178:180, 20:180] = 255
        white[20:180, 20:22] = 255
        white[20:180, 178:180] = 255
        ordered = np.array(
            [[20.0, 180.0], [180.0, 180.0], [180.0, 20.0], [20.0, 20.0]],
            dtype=np.float32,
        )
        frac = _quad_interior_white_fraction(white, ordered)
        self.assertLess(frac, MAX_QUAD_INTERIOR_WHITE)
        self.assertLess(frac, 0.08)

    def test_interior_fraction_rejects_a_filled_white_card(self):
        white = np.full((200, 200), 255, dtype=np.uint8)
        ordered = np.array(
            [[20.0, 180.0], [180.0, 180.0], [180.0, 20.0], [20.0, 20.0]],
            dtype=np.float32,
        )
        frac = _quad_interior_white_fraction(white, ordered)
        self.assertGreater(frac, MAX_QUAD_INTERIOR_WHITE)

    def test_detect_court_rejects_synthetic_bumper(self):
        frame = _bumper_frame()
        with self.assertRaises(RuntimeError):
            detect_court(frame)

    def test_detect_court_rejects_tennis_mp4_bumper(self):
        if not VIDEO.exists():
            self.skipTest(f"missing {VIDEO}")
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 2400)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.skipTest("could not read tennis.mp4 frame 2400")
        with self.assertRaises(RuntimeError):
            detect_court(frame)

    def test_detect_court_on_sample_frame_still_finds_the_court(self):
        if not FRAME_0.exists():
            self.skipTest(f"missing {FRAME_0}")
        frame = cv2.imread(str(FRAME_0))
        det = detect_court(frame)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        near_y = 0.5 * (float(det.corners[0, 1]) + float(det.corners[1, 1]))
        self.assertLess(far_y, 200.0, det.corners)
        self.assertGreater(near_y, 480.0, det.corners)
        np.testing.assert_allclose(det.corners, IMAGE_CORNERS, atol=5.0)


if __name__ == "__main__":
    unittest.main()
