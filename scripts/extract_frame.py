"""Save one video frame to debug/ for court-detect experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from io_video import open_video, save_nth_frame

VIDEO_PATH = ROOT / "tennis2.mp4"
OUT_PATH = ROOT / "debug" / "frame_0.jpg"
FRAME_INDEX = 0


def extract_frame(
    video_path: Path = VIDEO_PATH,
    out_path: Path = OUT_PATH,
    index: int = FRAME_INDEX,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = open_video(str(video_path))
    ok = save_nth_frame(cap, index, str(out_path))
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} from {video_path}")
    return out_path


def main() -> None:
    path = extract_frame()
    print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
