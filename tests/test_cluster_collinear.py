import unittest
from pathlib import Path

import numpy as np

from detect.court import _angle_and_rho, _cluster_collinear, detect_court


def _line_y_at_x(line, x: float) -> float:
    x1, y1, x2, y2 = map(float, line)
    if abs(x2 - x1) < 1e-6:
        return 0.5 * (y1 + y2)
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def _cluster_ys(lines: np.ndarray, x: float = 640.0) -> list[float]:
    merged, _ = _cluster_collinear(lines)
    return sorted(_line_y_at_x(line, x) for line in merged)


class TestClusterCollinearSpatial(unittest.TestCase):
    def test_collinear_far_baseline_segments_still_merge(self):
        # Two short pieces of the far baseline (y=164) must remain one line.
        lines = np.array(
            [
                [79.0, 164.0, 120.0, 164.0],
                [464.0, 164.0, 505.0, 164.0],
            ],
            dtype=np.float32,
        )
        merged, supports = _cluster_collinear(lines)
        self.assertEqual(len(merged), 1)
        self.assertGreater(supports[0], 80.0)
        self.assertAlmostEqual(_line_y_at_x(merged[0], 480.0), 164.0, delta=2.0)

    def test_does_not_merge_far_baseline_with_distant_ad_line(self):
        # Reproduces tennis2.mp4 frame 0: far-baseline segments at y=164 and a
        # ~3° ad line near y=90. Angle/rho are close enough to merge without a
        # spatial check; fitLine then tilts the baseline off the court.
        far_a = [79.0, 164.0, 120.0, 164.0]
        far_b = [464.0, 164.0, 505.0, 164.0]
        ad = [1181.0, 90.0, 1255.0, 86.0]
        lines = np.array([far_a, far_b, ad], dtype=np.float32)

        # Confirm the (angle, rho) pair would pass the old key check.
        a0, r0 = _angle_and_rho(np.array(far_a))
        a1, r1 = _angle_and_rho(np.array(ad))
        from detect.court import _angle_diff

        self.assertLess(_angle_diff(a0, a1), 4.0)
        self.assertLess(abs(r0 - r1), 20.0)

        ys = _cluster_ys(lines)
        self.assertGreaterEqual(len(ys), 2)
        self.assertTrue(any(abs(y - 164.0) < 5.0 for y in ys), ys)
        self.assertTrue(any(y < 130.0 for y in ys), ys)

    def test_detect_court_on_sample_frame_keeps_far_baseline(self):
        import cv2

        path = Path(__file__).resolve().parents[1] / "debug" / "frame_0.jpg"
        if not path.exists():
            self.skipTest(f"missing {path}")
        frame = cv2.imread(str(path))
        det = detect_court(frame)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        near_y = 0.5 * (float(det.corners[0, 1]) + float(det.corners[1, 1]))
        # Far baseline on this camera is ~y=164, not the service line (~y=240).
        self.assertLess(far_y, 200.0)
        self.assertGreater(near_y, 480.0)
        clicks = np.array(
            [[279.6, 502.8], [1002.4, 511.8], [824.7, 167.8], [458.3, 164.8]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(det.corners, clicks, atol=5.0)

    def test_detect_court_on_tennis2_frame0_does_not_use_service_line(self):
        import cv2

        path = Path(__file__).resolve().parents[1] / "tennis2.mp4"
        if not path.exists():
            self.skipTest(f"missing {path}")
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.skipTest("could not read frame 0")
        det = detect_court(frame)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        near_y = 0.5 * (float(det.corners[0, 1]) + float(det.corners[1, 1]))
        # Before the spatial cluster check this returned far_y≈239 (service line).
        self.assertLess(far_y, 200.0, det.corners)
        self.assertGreater(near_y, 480.0, det.corners)
        self.assertGreater(far_y, 130.0, det.corners)


if __name__ == "__main__":
    unittest.main()
