import unittest
from pathlib import Path

import numpy as np

from detect.court import _is_full_court_extent, detect_court

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

# detect_court(tennis2.mp4 frame 600): net used as the far "baseline".
# Homography would map the full court onto the near half (baseline → net).
NEAR_HALF_FRAME_600 = np.array(
    [
        [229.8, 548.2],
        [1055.4, 561.5],
        [948.1, 356.9],
        [338.1, 346.6],
    ],
    dtype=np.float32,
)

# Overlay on debug/court_frame_1500.jpg: net used as the near "baseline".
# Homography would map the full court onto the far half (net → far baseline).
FAR_HALF_FRAME_1500 = np.array(
    [
        [364.0, 271.0],
        [897.0, 271.0],
        [823.0, 164.0],
        [458.0, 164.0],
    ],
    dtype=np.float32,
)

FRAME_0 = Path(__file__).resolve().parents[1] / "debug" / "frame_0.jpg"


class TestFullCourtExtent(unittest.TestCase):
    def test_sample_clicks_span_full_court(self):
        self.assertTrue(_is_full_court_extent(IMAGE_CORNERS, FRAME_SHAPE))

    def test_rejects_near_half_where_net_is_far_edge(self):
        self.assertFalse(_is_full_court_extent(NEAR_HALF_FRAME_600, FRAME_SHAPE))

    def test_rejects_far_half_where_net_is_near_edge(self):
        self.assertFalse(_is_full_court_extent(FAR_HALF_FRAME_1500, FRAME_SHAPE))

    def test_detect_court_on_sample_frame_keeps_far_baseline(self):
        import cv2

        if not FRAME_0.exists():
            self.skipTest(f"missing {FRAME_0}")
        frame = cv2.imread(str(FRAME_0))
        det = detect_court(frame)
        self.assertTrue(_is_full_court_extent(det.corners, frame.shape))
        far_y = 0.5 * (det.corners[2, 1] + det.corners[3, 1])
        # Far baseline on this frame is ~y=164, not the net (~y=350).
        self.assertLess(float(far_y), 0.30 * frame.shape[0])
        np.testing.assert_allclose(det.corners, IMAGE_CORNERS, atol=5.0)


if __name__ == "__main__":
    unittest.main()
