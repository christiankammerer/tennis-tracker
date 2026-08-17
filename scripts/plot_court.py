"""Plot Court region polygons to debug/court_regions.png."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from court.model import Court

OUT_PATH = Path("debug/court_regions.png")

# property name → face color
REGION_COLORS = {
    "left_double_near": "#4e79a7",
    "right_double_near": "#f28e2b",
    "left_double_far": "#59a14f",
    "right_double_far": "#e15759",
}


def _as_ring(pts: np.ndarray) -> np.ndarray:
    """Sort rectangle corners into a non-crossing ring for filling."""
    pts = np.asarray(pts, dtype=float)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def plot_court(court: Court | None = None, out_path: Path = OUT_PATH) -> Path:
    court = court or Court()
    fig, ax = plt.subplots(figsize=(5, 9))

    # Outer doubles court
    outer = _as_ring(court.court_corners)
    ax.add_patch(
        Polygon(outer, closed=True, fill=False, edgecolor="black", linewidth=2, label="doubles court")
    )

    # Singles sidelines + net + service lines (guides)
    x0, x1 = court.singles_sideline, court.court_w - court.singles_sideline
    net_y = court.court_l / 2
    near_sl = net_y + court.sl_to_net
    far_sl = net_y - court.sl_to_net
    ax.plot([x0, x0], [0, court.court_l], color="gray", linestyle="--", linewidth=1, label="singles sideline")
    ax.plot([x1, x1], [0, court.court_l], color="gray", linestyle="--", linewidth=1)
    ax.plot([0, court.court_w], [net_y, net_y], color="black", linewidth=1.5, label="net")
    ax.plot([x0, x1], [near_sl, near_sl], color="gray", linewidth=1, label="service line")
    ax.plot([x0, x1], [far_sl, far_sl], color="gray", linewidth=1)
    ax.plot([court.court_w / 2, court.court_w / 2], [far_sl, near_sl], color="gray", linewidth=1, label="center line")

    for name, color in REGION_COLORS.items():
        pts = getattr(court, name)
        ring = _as_ring(pts)
        ax.add_patch(
            Polygon(ring, closed=True, facecolor=color, edgecolor="black", alpha=0.45, label=name)
        )
        c = ring.mean(axis=0)
        ax.text(c[0], c[1], name.replace("_", "\n"), ha="center", va="center", fontsize=7)

    ax.text(court.court_w / 2, court.court_l + 0.4, "NEAR (y = court_l)", ha="center", fontsize=9)
    ax.text(court.court_w / 2, -0.6, "FAR (y = 0)", ha="center", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, court.court_w + 0.5)
    ax.set_ylim(-1.2, court.court_l + 1.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Court region properties")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = plot_court()
    print(f"Wrote {path.resolve()}")
