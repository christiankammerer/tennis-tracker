"""detect_court must return doubles-pixel corners for Court.court_corners.

The Hough path locks onto the painted singles rectangle (blue on a US Open
broadcast). Pairing those pixels with Court.court_corners (doubles, 10.97 m)
stretches x by 10.97/8.23, so a ball 1 m inside the singles sideline maps
to x < 1.37 m and is called out.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from court.model import Court
from detect.court import _lift_singles_quad_to_doubles, detect_court

ROOT = Path(__file__).resolve().parents[1]
FRAME0 = ROOT / "debug" / "frame_0.jpg"

# Manual singles-court clicks from pipeline.py / choose_corners.py
SINGLES_CLICKS = np.float32(
    [
        [279.6, 502.8],
        [1002.4, 511.8],
        [824.7, 167.8],
        [458.3, 164.8],
    ]
)


def _image_to_court(H: np.ndarray, pixels) -> np.ndarray:
    pts = np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2)
    ok, H_inv = cv2.invert(np.asarray(H, dtype=np.float64))
    if not ok:
        raise AssertionError("H is not invertible")
    return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)


class TestSinglesVsDoublesHomography(unittest.TestCase):
    def setUp(self) -> None:
        self.court = Court()
        self.s = self.court.singles_sideline
        self.w = self.court.court_w
        self.L = self.court.court_l

    def test_singles_corners_are_inset_by_the_alley(self) -> None:
        corners = self.court.singles_corners
        np.testing.assert_allclose(corners[0], [self.s, self.L])
        np.testing.assert_allclose(corners[1], [self.w - self.s, self.L])
        np.testing.assert_allclose(corners[2], [self.w - self.s, 0.0])
        np.testing.assert_allclose(corners[3], [self.s, 0.0])
        self.assertAlmostEqual(float(self.w - 2 * self.s), 8.23, places=2)

    def test_lift_makes_court_corners_round_trip_to_singles_pixels(self) -> None:
        doubles_px = _lift_singles_quad_to_doubles(SINGLES_CLICKS, self.court)
        H = cv2.getPerspectiveTransform(
            self.court.court_corners.astype(np.float32), doubles_px
        )
        recovered = cv2.perspectiveTransform(
            self.court.singles_corners.astype(np.float32).reshape(-1, 1, 2), H
        ).reshape(4, 2)
        np.testing.assert_allclose(recovered, SINGLES_CLICKS, atol=0.05)

        singles_nl = _image_to_court(H, SINGLES_CLICKS[0])[0]
        np.testing.assert_allclose(singles_nl, [self.s, self.L], atol=1e-3)
        doubles_nl = _image_to_court(H, doubles_px[0])[0]
        np.testing.assert_allclose(doubles_nl, [0.0, self.L], atol=1e-3)

    def test_unlifted_singles_pixels_with_doubles_model_call_in_ball_out(self) -> None:
        # Current-bug reproduction without the lift: singles clicks + doubles
        # court_corners map a ball 1 m inside the singles sideline to x < 1.37.
        H_wrong = cv2.getPerspectiveTransform(
            self.court.court_corners.astype(np.float32), SINGLES_CLICKS
        )
        inside_1m = np.float32([[[self.s + 1.0, self.L]]])
        H_true = cv2.getPerspectiveTransform(
            self.court.singles_corners.astype(np.float32), SINGLES_CLICKS
        )
        pix = cv2.perspectiveTransform(inside_1m, H_true)
        mapped = _image_to_court(H_wrong, pix)[0]
        self.assertLess(float(mapped[0]), self.s)

        H_right = cv2.getPerspectiveTransform(
            self.court.court_corners.astype(np.float32),
            _lift_singles_quad_to_doubles(SINGLES_CLICKS, self.court),
        )
        mapped_ok = _image_to_court(H_right, pix)[0]
        np.testing.assert_allclose(mapped_ok, [self.s + 1.0, self.L], atol=1e-3)

    def test_detect_court_on_sample_frame_matches_doubles_model(self) -> None:
        if not FRAME0.exists():
            self.skipTest(f"missing {FRAME0}")
        frame = cv2.imread(str(FRAME0))
        self.assertIsNotNone(frame)
        det = detect_court(frame)
        H = cv2.getPerspectiveTransform(
            self.court.court_corners.astype(np.float32), det.corners
        )

        singles_px = cv2.perspectiveTransform(
            self.court.singles_corners.astype(np.float32).reshape(-1, 1, 2), H
        ).reshape(4, 2)
        # Projected singles corners stay on the painted blue rectangle
        # (manual clicks / previous detect_court output).
        np.testing.assert_allclose(singles_px, SINGLES_CLICKS, atol=8.0)

        nl = _image_to_court(H, det.corners[0])[0]
        np.testing.assert_allclose(nl, [0.0, self.L], atol=0.05)

        inside_1m = np.float32([[[self.s + 1.0, self.L]]])
        pix = cv2.perspectiveTransform(inside_1m, H)
        mapped = _image_to_court(H, pix)[0]
        np.testing.assert_allclose(mapped, [self.s + 1.0, self.L], atol=1e-3)
        self.assertGreater(float(mapped[0]), self.s)


if __name__ == "__main__":
    unittest.main()
