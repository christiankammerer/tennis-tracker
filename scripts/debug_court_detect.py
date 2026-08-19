"""Reproduce the court-detect debug images in debug/.

Run from the repo root:
    python scripts/extract_frame.py
    python scripts/debug_court_detect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect.court import (
    _angle_and_rho,
    _cluster_collinear,
    _is_baseline,
    _line_intersection,
    _order_corners,
    detect_court,
)

IMAGE_PATH = ROOT / "debug" / "frame_0.jpg"
OUT_DIR = ROOT / "debug"


def _load_frame() -> np.ndarray:
    frame = cv2.imread(str(IMAGE_PATH))
    if frame is None:
        raise FileNotFoundError(f"Could not read {IMAGE_PATH}. Run scripts/extract_frame.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return frame


def _write(name: str, image: np.ndarray) -> Path:
    path = OUT_DIR / name
    cv2.imwrite(str(path), image)
    print(f"Wrote {path}")
    return path


def _draw_quad(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    pts = corners.astype(int)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 136), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 136), -1)
        cv2.putText(
            vis,
            str(i + 1),
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return vis


def _draw_segments(frame: np.ndarray, lines: np.ndarray, label: bool = False) -> np.ndarray:
    vis = frame.copy()
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = map(int, line)
        angle, _ = _angle_and_rho(line)
        color = (0, 255, 136) if _is_baseline(angle) else (0, 128, 255)
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)
        if label:
            mx, my = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
            cv2.putText(vis, str(i), (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return vis


def _legacy_white_lines(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(white, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=20,
    )
    if lines is None:
        raise RuntimeError("No court lines found")
    return np.asarray(lines).reshape(-1, 4)


def _legacy_cluster(lines: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    return _cluster_collinear(lines, angle_tol=8.0, rho_tol=30.0)


def _valid_quads(
    merged: list[np.ndarray],
    supports: list[float],
    shape: tuple[int, ...],
    min_support: float = 0.0,
) -> list[tuple[float, float, np.ndarray]]:
    height, width = shape[:2]
    baselines, sidelines = [], []
    for line, support in zip(merged, supports):
        if support < min_support:
            continue
        angle, _ = _angle_and_rho(line)
        rec = (line, support)
        if _is_baseline(angle):
            baselines.append(rec)
        else:
            sidelines.append(rec)

    def inside(pt: np.ndarray) -> bool:
        x, y = float(pt[0]), float(pt[1])
        return -80 <= x < width + 80 and -80 <= y < height + 80

    cands: list[tuple[float, float, np.ndarray]] = []
    for i, (b1, s1) in enumerate(baselines):
        for j, (b2, s2) in enumerate(baselines):
            if j <= i:
                continue
            for k, (l1, sl1) in enumerate(sidelines):
                for m, (l2, sl2) in enumerate(sidelines):
                    if m <= k:
                        continue
                    try:
                        pts = np.stack([
                            _line_intersection(b1, l1),
                            _line_intersection(b1, l2),
                            _line_intersection(b2, l2),
                            _line_intersection(b2, l1),
                        ])
                    except ValueError:
                        continue
                    if not all(inside(p) for p in pts):
                        continue
                    ordered = _order_corners(pts, shape)
                    near_left, near_right, far_right, far_left = ordered
                    near_w = float(np.linalg.norm(near_right - near_left))
                    far_w = float(np.linalg.norm(far_right - far_left))
                    box_h = abs(
                        0.5 * (near_left[1] + near_right[1]) - 0.5 * (far_left[1] + far_right[1])
                    )
                    if near_w < 50 or far_w < 50 or box_h < 50 or near_w < far_w:
                        continue
                    area = float(cv2.contourArea(ordered.astype(np.float32)))
                    cands.append((s1 + s2 + sl1 + sl2, area, ordered))
    return cands


def write_color_masks(frame: np.ndarray) -> None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 30, 50), (130, 255, 220))
    green = cv2.inRange(hsv, (40, 30, 50), (90, 255, 220))
    surface = cv2.bitwise_or(blue, green)
    surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    dilated = cv2.dilate(surface, np.ones((15, 15), np.uint8))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    white_on = cv2.bitwise_and(white, dilated)

    _write("mask_blue.jpg", blue)
    _write("mask_green.jpg", green)
    _write("mask_surface.jpg", surface)
    _write("mask_white.jpg", white)
    _write("mask_white_on_court.jpg", white_on)

    vis = frame.copy()
    edges = cv2.Canny(white_on, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=60, maxLineGap=20)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 136), 2)
    _write("hough_on_court.jpg", vis)


def write_blue_dilated_masks(frame: np.ndarray) -> None:
    height = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 30, 40), (130, 255, 230))
    blue = cv2.dilate(blue, np.ones((21, 21), np.uint8))
    blue[: int(0.12 * height)] = 0
    blue[int(0.90 * height) :] = 0
    _write("mask_blue_dilated.jpg", blue)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    white_on = cv2.bitwise_and(white, blue)
    white_on = cv2.morphologyEx(white_on, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _write("mask_white_blue.jpg", white_on)


def write_hough_raw(frame: np.ndarray) -> None:
    lines = _legacy_white_lines(frame)
    _write("hough_raw.jpg", _draw_segments(frame, lines, label=True))


def write_hough_sensitive_horiz(frame: np.ndarray) -> None:
    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white[int(0.90 * height) :] = 0
    edges = cv2.Canny(white, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=25)
    vis = frame.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = abs(np.degrees(np.atan2(y2 - y1, x2 - x1))) % 180
            if min(angle, 180 - angle) < 20:
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 136), 2)
    _write("hough_sensitive_horiz.jpg", vis)


def write_legacy_quads(frame: np.ndarray) -> None:
    lines = _legacy_white_lines(frame)
    merged, supports = _legacy_cluster(lines)
    cands = _valid_quads(merged, supports, frame.shape)
    if not cands:
        raise RuntimeError("No valid legacy quads")

    by_support = max(cands, key=lambda c: c[0])
    by_area = max(cands, key=lambda c: c[1])
    _write("court_best_support.jpg", _draw_quad(frame, by_support[2]))
    _write("court_max_area.jpg", _draw_quad(frame, by_area[2]))


def write_court_detect(frame: np.ndarray) -> None:
    det = detect_court(frame)
    print("corners (near-left, near-right, far-right, far-left):")
    print(det.corners)
    _write("court_detect.jpg", _draw_quad(frame, det.corners))


def main() -> None:
    frame = _load_frame()
    write_color_masks(frame)
    write_blue_dilated_masks(frame)
    write_hough_raw(frame)
    write_hough_sensitive_horiz(frame)
    write_legacy_quads(frame)
    write_court_detect(frame)


if __name__ == "__main__":
    main()
