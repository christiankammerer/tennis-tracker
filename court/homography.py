import cv2
import numpy as np
import json


def build_H(corners_src, corners_dst):
    """
    Build the homography matrix H using the corners of the court and the destination images.
    """
    return cv2.getPerspectiveTransform(corners_src, corners_dst)

def court_to_image(H, court_corners):
    """
    Convert the court corners to image corners using the homography matrix H.
    """
    return cv2.perspectiveTransform(court_corners, H)

def image_to_court(H, image_corners):
    """
    Convert the image corners to court corners using the homography matrix H.
    """
    return cv2.perspectiveTransform(image_corners, H)

def save_calibration(image_path, image_corners, court_corners, H, filename="calibration.json"):
  with open(filename, "w") as f:
    json.dump({
      "image_path": image_path,
      "image_corners": image_corners.tolist(),
      "court_corners": court_corners.tolist(),
      "H": H.tolist(),
    }, f)

def load_calibration(filename="calibration.json"):
  with open(filename, "r") as f:
    return json.load(f)