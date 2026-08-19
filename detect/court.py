from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from court.model import Court

CORNER_ORDER = ("near_left", "near_right", "far_right", "far_left")


@dataclass
class CourtDetection:
    corners: np.ndarray
    confidence: float = 1.0


def detect_court(frame: np.ndarray) -> CourtDetection:
    """Return the 4 outer court corners in pixel coordinates.
    Order: near-left, near-right, far-right, far-left.
    """
    white, lines = _find_white_lines(frame)
    raw_corners = _intersect_outer_lines(lines, frame.shape, white)
    corners = _order_corners(raw_corners, frame.shape)
    return CourtDetection(corners=corners, confidence=1.0)


def _find_white_lines(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a white-line mask and Hough segments as (N, 4)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    height = white.shape[0]
    white[: int(0.10 * height), :] = 0
    white[int(0.90 * height) :, :] = 0
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    edges = cv2.Canny(white, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=25,
    )
    if lines is None:
        raise RuntimeError("No court lines found")
    return white, np.asarray(lines).reshape(-1, 4)


def _angle_and_rho(line: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, line)
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-6:
        raise ValueError("zero-length segment")

    angle = np.degrees(np.atan2(dy, dx)) % 180.0

    nx, ny = -dy, dx
    if nx < 0 or (nx == 0 and ny < 0):
        nx, ny = -nx, -ny
    rho = (x1 * nx + y1 * ny) / length
    return angle, rho


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _is_baseline(angle: float) -> bool:
    return _angle_diff(angle, 0.0) < 25.0


def _segment_length(line: np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, line)
    return float(np.hypot(x2 - x1, y2 - y1))


def _cluster_collinear(
    lines: np.ndarray,
    angle_tol: float = 4.0,
    rho_tol: float = 20.0,
) -> tuple[list[np.ndarray], list[float]]:
    groups: list[list[np.ndarray]] = []
    keys: list[tuple[float, float]] = []
    supports: list[float] = []

    for line in lines:
        angle, rho = _angle_and_rho(line)
        placed = False
        for i, (ga, gr) in enumerate(keys):
            if _angle_diff(angle, ga) < angle_tol and abs(rho - gr) < rho_tol:
                groups[i].append(line)
                supports[i] += _segment_length(line)
                placed = True
                break
        if not placed:
            groups.append([line])
            keys.append((angle, rho))
            supports.append(_segment_length(line))

    merged = []
    for group in groups:
        pts = np.asarray(group, dtype=np.float32).reshape(-1, 2)
        vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        s = 10000.0
        merged.append(np.array([x0 - vx * s, y0 - vy * s, x0 + vx * s, y0 + vy * s]))
    return merged, supports


def _model_segments(court: Court) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    width, length = court.court_w, court.court_l
    singles = court.singles_sideline
    net = length / 2
    service = court.sl_to_net
    return [
        ((0.0, 0.0), (width, 0.0)),
        ((0.0, length), (width, length)),
        ((0.0, 0.0), (0.0, length)),
        ((width, 0.0), (width, length)),
        ((singles, 0.0), (singles, length)),
        ((width - singles, 0.0), (width - singles, length)),
        ((0.0, net), (width, net)),
        ((singles, net + service), (width - singles, net + service)),
        ((singles, net - service), (width - singles, net - service)),
        ((width / 2, net - service), (width / 2, net + service)),
    ]


def _model_support(H: np.ndarray, white: np.ndarray, court: Court, samples: int = 40) -> float:
    height, width = white.shape
    score = 0.0
    for p0, p1 in _model_segments(court):
        court_pts = np.array([[p0, p1]], dtype=np.float32)
        img_pts = cv2.perspectiveTransform(court_pts, H)[0]
        xs = np.linspace(img_pts[0, 0], img_pts[1, 0], samples)
        ys = np.linspace(img_pts[0, 1], img_pts[1, 1], samples)
        hits = 0
        for x, y in zip(xs, ys):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                if white[max(0, yi - 2) : yi + 3, max(0, xi - 2) : xi + 3].any():
                    hits += 1
        score += hits / samples
    return score


def _point_in_frame(pt: np.ndarray, shape: tuple[int, ...], margin: float = 40.0) -> bool:
    height, width = shape[:2]
    x, y = float(pt[0]), float(pt[1])
    return -margin <= x < width + margin and -margin <= y < height + margin


def _intersect_outer_lines(
    lines: np.ndarray,
    shape: tuple[int, ...],
    white: np.ndarray,
) -> np.ndarray:
    merged, supports = _cluster_collinear(lines)

    baselines, sidelines = [], []
    for line, support in zip(merged, supports):
        if support < 40.0:
            continue
        angle, _ = _angle_and_rho(line)
        if _is_baseline(angle):
            baselines.append(line)
        else:
            sidelines.append(line)

    if len(baselines) < 2 or len(sidelines) < 2:
        raise RuntimeError(
            f"Need 2 baselines and 2 sidelines, got {len(baselines)} / {len(sidelines)}"
        )

    court = Court()
    court_src = court.court_corners.astype(np.float32)
    best_score = -1.0
    best_corners: np.ndarray | None = None

    for i, b1 in enumerate(baselines):
        for j, b2 in enumerate(baselines):
            if j <= i:
                continue
            for k, s1 in enumerate(sidelines):
                for m, s2 in enumerate(sidelines):
                    if m <= k:
                        continue
                    try:
                        pts = np.stack([
                            _line_intersection(b1, s1),
                            _line_intersection(b1, s2),
                            _line_intersection(b2, s2),
                            _line_intersection(b2, s1),
                        ])
                    except ValueError:
                        continue
                    if not all(np.isfinite(p).all() and _point_in_frame(p, shape) for p in pts):
                        continue
                    ordered = _order_corners(pts, shape)
                    near_left, near_right, far_right, far_left = ordered
                    near_w = float(np.linalg.norm(near_right - near_left))
                    far_w = float(np.linalg.norm(far_right - far_left))
                    height = abs(
                        0.5 * (near_left[1] + near_right[1]) - 0.5 * (far_left[1] + far_right[1])
                    )
                    if near_w < 80 or far_w < 40 or height < 80 or near_w < far_w:
                        continue
                    try:
                        H = cv2.getPerspectiveTransform(court_src, ordered)
                    except cv2.error:
                        continue
                    score = _model_support(H, white, court)
                    if score > best_score:
                        best_score = score
                        best_corners = ordered

    if best_corners is None:
        raise RuntimeError("No court quadrilateral matched the court model")
    return best_corners


def _line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Intersect two segments given as (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    p1 = np.array([x1, y1, 1.0])
    p2 = np.array([x2, y2, 1.0])
    p3 = np.array([x3, y3, 1.0])
    p4 = np.array([x4, y4, 1.0])
    l_a = np.cross(p1, p2)
    l_b = np.cross(p3, p4)
    pt = np.cross(l_a, l_b)
    if abs(pt[2]) < 1e-8:
        raise ValueError("Lines are parallel")
    return (pt[:2] / pt[2]).astype(np.float32)


def _order_corners(pts: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sort 4 points into near-left, near-right, far-right, far-left.

    In image space, 'near' means larger y (bottom of the frame).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    by_y = pts[np.argsort(pts[:, 1])]
    far = by_y[:2]
    near = by_y[2:]
    far = far[np.argsort(far[:, 0])]
    near = near[np.argsort(near[:, 0])]
    ordered = np.stack([near[0], near[1], far[1], far[0]])
    return ordered.astype(np.float32)
