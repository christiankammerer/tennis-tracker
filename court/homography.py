import cv2
import numpy as np
import json


def _as_xy(points):
    """Convert point-like input to float32 (N, 2). OpenCV 5 requires CV_32F."""
    pts = np.asarray(points, dtype=np.float32)
    return pts.reshape(-1, 2)


def build_H(corners_src, corners_dst):
    """
    Build the homography matrix H using the corners of the court and the destination images.

    H maps source (court-plane) points to destination (image) points.
    """
    src = _as_xy(corners_src)
    dst = _as_xy(corners_dst)
    return cv2.getPerspectiveTransform(src, dst)


def court_to_image(H, court_corners):
    """
    Convert court-plane points to image pixels using H (court → image).
    """
    pts = _as_xy(court_corners).reshape(-1, 1, 2)
    H = np.asarray(H, dtype=np.float64)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def image_to_court(H, image_corners):
    """
    Convert image pixels to court-plane points using H (court → image).

    Applies H^{-1}; H is the matrix returned by build_H(court_corners, image_corners).
    """
    pts = _as_xy(image_corners).reshape(-1, 1, 2)
    H = np.asarray(H, dtype=np.float64)
    ok, H_inv = cv2.invert(H)
    if not ok:
        raise ValueError("Homography matrix H is not invertible")
    return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)


def save_calibration(image_path, image_corners, court_corners, H, filename="calibration.json"):
  with open(filename, "w") as f:
    json.dump({
      "image_path": image_path,
      "image_corners": np.asarray(image_corners).tolist(),
      "court_corners": np.asarray(court_corners).tolist(),
      "H": np.asarray(H).tolist(),
    }, f)


def load_calibration(filename="calibration.json"):
  with open(filename, "r") as f:
    return json.load(f)
