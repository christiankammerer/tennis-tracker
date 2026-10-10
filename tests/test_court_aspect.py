import unittest
from pathlib import Path

import numpy as np

from detect.court import _is_full_court_aspect, detect_court

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

# detect_court(tennis2.mp4 frame 450) before the aspect filter.
# Far edge is the far service line (y≈246), not the far baseline (y≈160).
# Homography would map 23.77 m onto ~18.3 m of court.
FRAME_450_SERVICE = np.array(
    [
        [209.6158, 557.1880],
        [1063.2679, 566.0658],
        [896.1708, 250.2066],
        [378.1684, 242.4175],
    ],
    dtype=np.float32,
)

# Same frame, same sidelines, but paired with the true far baseline.
FRAME_450_FULL = np.array(
    [
        [209.6158, 557.1880],
        [1063.2679, 566.0658],
        [848.4493, 160.0],
        [422.3011, 160.0],
    ],
    dtype=np.float32,
)

VIDEO = Path(__file__).resolve().parents[1] / "tennis2.mp4"
FRAME_0 = Path(__file__).resolve().parents[1] / "debug" / "frame_0.jpg"


class TestFullCourtAspect(unittest.TestCase):
    def test_sample_clicks_have_full_court_aspect(self):
        self.assertTrue(_is_full_court_aspect(IMAGE_CORNERS))

    def test_rejects_far_service_line_as_far_baseline(self):
        self.assertFalse(_is_full_court_aspect(FRAME_450_SERVICE))

    def test_accepts_true_far_baseline_on_same_frame(self):
        self.assertTrue(_is_full_court_aspect(FRAME_450_FULL))

    def test_detect_court_frame_450_uses_far_baseline(self):
        import cv2

        if not VIDEO.exists():
            self.skipTest(f"missing {VIDEO}")
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 450)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.skipTest("could not read frame 450")
        det = detect_court(frame)
        self.assertTrue(_is_full_court_aspect(det.corners), det.corners)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
        near_y = 0.5 * (float(det.corners[0, 1]) + float(det.corners[1, 1]))
        # Before the filter this returned far_y≈246 (service line).
        self.assertLess(far_y, 200.0, det.corners)
        self.assertGreater(near_y, 520.0, det.corners)
        np.testing.assert_allclose(det.corners, FRAME_450_FULL, atol=3.0)

    def test_detect_court_on_sample_frame_keeps_far_baseline(self):
        import cv2

        if not FRAME_0.exists():
            self.skipTest(f"missing {FRAME_0}")
        frame = cv2.imread(str(FRAME_0))
        det = detect_court(frame)
        self.assertTrue(_is_full_court_aspect(det.corners), det.corners)
        far_y = 0.5 * (float(det.corners[2, 1]) + float(det.corners[3, 1]))
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
        self.assertLess(far_y, 200.0, det.corners)
        self.assertGreater(near_y, 520.0, det.corners)
        self.assertTrue(_is_full_court_aspect(det.corners), det.corners)


if __name__ == "__main__":
    unittest.main()
