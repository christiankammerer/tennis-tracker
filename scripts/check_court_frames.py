"""Run detect_court on several frames from the video.

Writes debug/court_frame_{index}.jpg for each sample.
Run from the repo root:
    python scripts/check_court_frames.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect.court import detect_court
from io_video import frame_count, open_video


def _draw_quad(frame, corners):
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

VIDEO_PATH = ROOT / "tennis2.mp4"
OUT_DIR = ROOT / "debug"
SAMPLE_EVERY_SEC = 10


def sample_indices(n_frames: int, fps: float, every_sec: float = SAMPLE_EVERY_SEC) -> list[int]:
    step = max(int(round(fps * every_sec)), 1)
    return list(range(0, n_frames, step))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = open_video(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {VIDEO_PATH}")

    n_frames = frame_count(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    indices = sample_indices(n_frames, fps)
    print(f"{VIDEO_PATH.name}: {n_frames} frames @ {fps:.1f} fps")
    print(f"testing {len(indices)} frames: {indices}")

    ok = 0
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            print(f"frame {index}: could not read")
            continue
        try:
            det = detect_court(frame)
        except Exception as exc:
            print(f"frame {index}: FAIL {exc}")
            continue
        out = OUT_DIR / f"court_frame_{index}.jpg"
        cv2.imwrite(str(out), _draw_quad(frame, det.corners))
        pts = [f"({x:.1f}, {y:.1f})" for x, y in det.corners]
        print(f"frame {index}: {' '.join(pts)} -> {out.name}")
        ok += 1

    cap.release()
    print(f"{ok}/{len(indices)} succeeded")


if __name__ == "__main__":
    main()
