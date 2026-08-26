import unittest
from pathlib import Path

import numpy as np

from detect.court import _is_valid_outer_quad, _order_corners, detect_court

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

# detect_court(tennis2.mp4 frame 1800) before the degenerate-quad filter.
# near-left and far-left are 1px apart; H maps the court center onto the baseline.
DEGENERATE_FRAME_1800 = np.array(
    [
        [292.5, 627.3],
        [1075.2, 647.3],
        [933.0, 377.1],
        [292.9, 626.4],
    ],
    dtype=np.float32,
)

FRAME_0 = Path(__file__).resolve().parents[1] / "debug" / "frame_0.jpg"


class TestValidOuterQuad(unittest.TestCase):
    def test_sample_clicks_are_valid(self):
        self.assertTrue(_is_valid_outer_quad(IMAGE_CORNERS))

    def test_rejects_collapsed_sideline(self):
        self.assertFalse(_is_valid_outer_quad(DEGENERATE_FRAME_1800))

    def test_rejects_duplicate_corner_after_ordering(self):
        pts = np.array(
            [
                [200.0, 500.0],
                [900.0, 500.0],
                [700.0, 150.0],
                [200.0, 500.0],
            ],
            dtype=np.float32,
        )
        ordered = _order_corners(pts, (720, 1280))
        self.assertFalse(_is_valid_outer_quad(ordered))

    def test_detect_court_on_sample_frame_matches_clicks(self):
        import cv2

        if not FRAME_0.exists():
            self.skipTest(f"missing {FRAME_0}")
        frame = cv2.imread(str(FRAME_0))
        det = detect_court(frame)
        self.assertTrue(_is_valid_outer_quad(det.corners))
        np.testing.assert_allclose(det.corners, IMAGE_CORNERS, atol=5.0)


if __name__ == "__main__":
    unittest.main()
