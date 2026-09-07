import unittest

import cv2
import numpy as np

from court.model import Court


class TestDoublesAlleys(unittest.TestCase):
    def setUp(self):
        self.court = Court()
        self.net = self.court.court_l / 2
        self.expected_area = self.court.singles_sideline * (self.court.court_l / 2)

    def test_near_alleys_sit_on_the_near_half(self):
        # A bounce at (0.5 m, 20 m) is in the near-left alley (near baseline is y=23.77).
        # The old properties placed left_double_near on y=0..net (the far half).
        for name in ("left_double_near", "right_double_near"):
            ys = getattr(self.court, name)[:, 1]
            self.assertGreaterEqual(float(ys.min()), self.net - 1e-6, name)
            self.assertAlmostEqual(float(ys.max()), self.court.court_l, places=6)

    def test_far_alleys_sit_on_the_far_half(self):
        for name in ("left_double_far", "right_double_far"):
            ys = getattr(self.court, name)[:, 1]
            self.assertLessEqual(float(ys.max()), self.net + 1e-6, name)
            self.assertAlmostEqual(float(ys.min()), 0.0, places=6)

    def test_alley_quads_are_non_crossing_rectangles(self):
        # Previous vertex order was a bowtie (contourArea == 0).
        for name in (
            "left_double_near",
            "right_double_near",
            "left_double_far",
            "right_double_far",
        ):
            pts = getattr(self.court, name).astype(np.float32)
            area = float(cv2.contourArea(pts))
            self.assertGreater(area, 1.0, name)
            self.assertAlmostEqual(area, self.expected_area, places=4, msg=name)

    def test_point_in_polygon_matches_named_end(self):
        left_near = self.court.left_double_near.astype(np.float32).reshape(-1, 1, 2)
        left_far = self.court.left_double_far.astype(np.float32).reshape(-1, 1, 2)
        right_near = self.court.right_double_near.astype(np.float32).reshape(-1, 1, 2)

        near_left_bounce = (0.5, 20.0)
        far_left_bounce = (0.5, 2.0)
        near_right_bounce = (10.5, 20.0)

        self.assertGreater(cv2.pointPolygonTest(left_near, near_left_bounce, False), 0)
        self.assertLess(cv2.pointPolygonTest(left_far, near_left_bounce, False), 0)
        self.assertGreater(cv2.pointPolygonTest(left_far, far_left_bounce, False), 0)
        self.assertLess(cv2.pointPolygonTest(left_near, far_left_bounce, False), 0)
        self.assertGreater(cv2.pointPolygonTest(right_near, near_right_bounce, False), 0)
        self.assertLess(cv2.pointPolygonTest(left_near, near_right_bounce, False), 0)


if __name__ == "__main__":
    unittest.main()
