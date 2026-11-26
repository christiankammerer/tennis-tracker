import unittest
from pathlib import Path

import cv2
import numpy as np

from detect.court import (
    MIN_AREA_FRAC,
    MIN_HEIGHT_FRAC,
    MIN_NEAR_WIDTH_FRAC,
    _is_plausible_outer_quad,
    detect_court,
)

FRAME_0 = Path(__file__).resolve().parents[1] / "debug" / "frame_0.jpg"
VIDEO = Path(__file__).resolve().parents[1] / "tennis.mp4"
VIDEO2 = Path(__file__).resolve().parents[1] / "tennis2.mp4"

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

# tennis.mp4 frame 2340: ~80×110 rectangle on Djokovic's shirt during a close-up.
SHIRT_CORNERS = np.array(
    [
        [439.0, 533.1],
        [519.2, 531.6],
        [508.6, 423.3],
        [439.0, 423.6],
    ],
    dtype=np.float32,
)

# Next-best candidate on the same frame after the shirt quad is filtered:
# a ~470×120 trapezoid spanning the torso / raised arms.
TORSO_CORNERS = np.array(
    [
        [413.0, 507.0],
        [887.0, 517.0],
        [876.0, 387.0],
        [509.0, 392.0],
    ],
    dtype=np.float32,
)

SHAPE_720P = (720, 1280, 3)


def _read_frame(path: Path, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


class TestCloseupQuad(unittest.TestCase):
    def test_shirt_sized_quad_is_rejected(self):
        self.assertFalse(_is_plausible_outer_quad(SHIRT_CORNERS, SHAPE_720P))
        near_w = float(np.linalg.norm(SHIRT_CORNERS[1] - SHIRT_CORNERS[0]))
        self.assertLess(near_w, MIN_NEAR_WIDTH_FRAC * SHAPE_720P[1])

    def test_torso_sized_quad_is_rejected(self):
        self.assertFalse(_is_plausible_outer_quad(TORSO_CORNERS, SHAPE_720P))
        height = abs(
            0.5 * (TORSO_CORNERS[0, 1] + TORSO_CORNERS[1, 1])
            - 0.5 * (TORSO_CORNERS[2, 1] + TORSO_CORNERS[3, 1])
        )
        area = float(cv2.contourArea(TORSO_CORNERS))
        self.assertLess(height, MIN_HEIGHT_FRAC * SHAPE_720P[0])
        self.assertLess(area, MIN_AREA_FRAC * SHAPE_720P[0] * SHAPE_720P[1])

    def test_sample_court_corners_still_pass(self):
        self.assertTrue(_is_plausible_outer_quad(IMAGE_CORNERS, SHAPE_720P))
        near_w = float(np.linalg.norm(IMAGE_CORNERS[1] - IMAGE_CORNERS[0]))
        height = abs(
            0.5 * (IMAGE_CORNERS[0, 1] + IMAGE_CORNERS[1, 1])
            - 0.5 * (IMAGE_CORNERS[2, 1] + IMAGE_CORNERS[3, 1])
        )
        self.assertGreater(near_w, MIN_NEAR_WIDTH_FRAC * SHAPE_720P[1])
        self.assertGreater(height, MIN_HEIGHT_FRAC * SHAPE_720P[0])

    def test_detect_court_rejects_celebration_closeup(self):
        if not VIDEO.exists():
            self.skipTest(f"missing {VIDEO}")
        frame = _read_frame(VIDEO, 2340)
        if frame is None:
            self.skipTest("could not read tennis.mp4 frame 2340")
        with self.assertRaises(RuntimeError):
            detect_court(frame)

    def test_detect_court_rejects_logo_quad_on_closeup(self):
        if not VIDEO.exists():
            self.skipTest(f"missing {VIDEO}")
        frame = _read_frame(VIDEO, 2360)
        if frame is None:
            self.skipTest("could not read tennis.mp4 frame 2360")
        with self.assertRaises(RuntimeError):
            detect_court(frame)

    def test_detect_court_on_sample_frame_still_finds_the_court(self):
        if not FRAME_0.exists():
            self.skipTest(f"missing {FRAME_0}")
        frame = cv2.imread(str(FRAME_0))
        det = detect_court(frame)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        near_y = 0.5 * (float(det.corners[0, 1]) + float(det.corners[1, 1]))
        near_w = float(np.linalg.norm(det.corners[1] - det.corners[0]))
        self.assertLess(far_y, 200.0, det.corners)
        self.assertGreater(near_y, 480.0, det.corners)
        self.assertGreater(near_w, MIN_NEAR_WIDTH_FRAC * frame.shape[1])
        np.testing.assert_allclose(det.corners, IMAGE_CORNERS, atol=5.0)

    def test_detect_court_on_tennis2_frame_300_still_wide(self):
        if not VIDEO2.exists():
            self.skipTest(f"missing {VIDEO2}")
        frame = _read_frame(VIDEO2, 300)
        if frame is None:
            self.skipTest("could not read tennis2.mp4 frame 300")
        det = detect_court(frame)
        near_w = float(np.linalg.norm(det.corners[1] - det.corners[0]))
        self.assertGreater(near_w, MIN_NEAR_WIDTH_FRAC * frame.shape[1], det.corners)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        self.assertLess(far_y, 220.0, det.corners)


if __name__ == "__main__":
    unittest.main()
