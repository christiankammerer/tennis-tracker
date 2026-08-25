import unittest

import numpy as np

from court.homography import build_H, court_to_image, image_to_court, save_calibration, load_calibration
from court.model import Court

# Pixel corners from the sample frame (near-left, near-right, far-right, far-left).
IMAGE_CORNERS = [
    (279.6, 502.8),
    (1002.4, 511.8),
    (824.7, 167.8),
    (458.3, 164.8),
]


class TestHomography(unittest.TestCase):
    def setUp(self):
        self.court = Court()
        self.image_corners = IMAGE_CORNERS

    def test_build_h_accepts_float64_court_corners_and_tuple_pixels(self):
        H = build_H(self.court.court_corners, self.image_corners)
        self.assertEqual(H.shape, (3, 3))
        self.assertTrue(np.isfinite(H).all())

    def test_court_to_image_maps_court_corners_to_pixels(self):
        H = build_H(self.court.court_corners, self.image_corners)
        pixels = court_to_image(H, self.court.court_corners)
        np.testing.assert_allclose(pixels, np.float32(self.image_corners), atol=1e-3)

    def test_image_to_court_inverts_h_to_recover_court_meters(self):
        H = build_H(self.court.court_corners, self.image_corners)
        recovered = image_to_court(H, self.image_corners)
        np.testing.assert_allclose(recovered, self.court.court_corners, atol=1e-4)

    def test_round_trip_interior_point(self):
        H = build_H(self.court.court_corners, self.image_corners)
        pts = np.array(
            [
                [self.court.court_w / 2, self.court.court_l / 2],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        )
        pixels = court_to_image(H, pts)
        back = image_to_court(H, pixels)
        np.testing.assert_allclose(back, pts, atol=1e-4)

    def test_image_to_court_is_not_applying_forward_h(self):
        H = build_H(self.court.court_corners, self.image_corners)
        recovered = image_to_court(H, self.image_corners)
        # Applying H (court→image) to image pixels produces huge garbage coords.
        self.assertTrue(np.all(np.abs(recovered) < 50.0))
        np.testing.assert_allclose(recovered[0], [0.0, self.court.court_l], atol=1e-4)

    def test_save_calibration_accepts_tuple_corners(self):
        import os
        import tempfile

        H = build_H(self.court.court_corners, self.image_corners)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "calibration.json")
            save_calibration("debug/frame_0.jpg", self.image_corners, self.court.court_corners, H, path)
            data = load_calibration(path)
            self.assertEqual(len(data["image_corners"]), 4)
            recovered = image_to_court(data["H"], data["image_corners"])
            np.testing.assert_allclose(recovered, self.court.court_corners, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
