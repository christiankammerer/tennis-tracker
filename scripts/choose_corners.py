from __future__ import annotations

import base64
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

"""Interactive court-corner picker (browser-based; works on WSL without Qt)."""

corners = [(279.6, 502.8), (1002.4, 511.8), (824.7, 167.8), (458.3, 164.8)]

IMAGE_PATH = Path("debug/frame_0.jpg")
CORNER_LABELS = (
    "near-left (bottom-left of court)",
    "near-right (bottom-right of court)",
    "far-right (top-right of court)",
    "far-left (top-left of court)",
)


def click_court_corners(image_bgr: np.ndarray, port: int = 8765) -> list[tuple[float, float]]:
    """Open a browser window; click 4 court corners in order. Returns pixel (x, y)."""
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for picker")
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    h, w = image_bgr.shape[:2]

    result: dict[str, list[tuple[float, float]] | None] = {"corners": None}
    ready = threading.Event()

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Court corner picker</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #111; color: #eee; }}
    #status {{ margin-bottom: 0.75rem; font-size: 1.1rem; }}
    #wrap {{ display: inline-block; position: relative; cursor: crosshair; }}
    canvas {{ max-width: 100%; height: auto; border: 1px solid #444; }}
    button {{ margin-right: 0.5rem; padding: 0.4rem 0.8rem; }}
  </style>
</head>
<body>
  <div id="status">Click corner 1/4: {CORNER_LABELS[0]}</div>
  <div>
    <button type="button" id="undo">Undo</button>
    <button type="button" id="reset">Reset</button>
  </div>
  <div id="wrap"><canvas id="c"></canvas></div>
  <script>
    const labels = {json.dumps(list(CORNER_LABELS))};
    const imgW = {w}, imgH = {h};
    const img = new Image();
    img.src = "data:image/jpeg;base64,{img_b64}";
    const canvas = document.getElementById("c");
    const ctx = canvas.getContext("2d");
    const status = document.getElementById("status");
    let corners = [];

    function draw() {{
      canvas.width = imgW;
      canvas.height = imgH;
      ctx.drawImage(img, 0, 0);
      corners.forEach((p, i) => {{
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = "#00ff88";
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.font = "bold 16px sans-serif";
        ctx.fillText(String(i + 1), p.x + 12, p.y - 8);
      }});
      if (corners.length >= 2) {{
        ctx.beginPath();
        ctx.moveTo(corners[0].x, corners[0].y);
        for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i].x, corners[i].y);
        if (corners.length === 4) ctx.closePath();
        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
    }}

    function updateStatus() {{
      if (corners.length < 4) {{
        status.textContent = `Click corner ${{corners.length + 1}}/4: ${{labels[corners.length]}}`;
      }} else {{
        status.textContent = "Done — sending coordinates…";
      }}
    }}

    function submit() {{
      const q = corners.map(p => `${{p.x.toFixed(1)}},${{p.y.toFixed(1)}}`).join(";");
      fetch("/done?corners=" + encodeURIComponent(q)).then(() => {{
        status.textContent = "Saved. You can close this tab.";
      }});
    }}

    img.onload = () => {{ draw(); updateStatus(); }};

    canvas.addEventListener("click", (e) => {{
      if (corners.length >= 4) return;
      const r = canvas.getBoundingClientRect();
      const x = (e.clientX - r.left) * (imgW / r.width);
      const y = (e.clientY - r.top) * (imgH / r.height);
      corners.push({{ x, y }});
      draw();
      updateStatus();
      if (corners.length === 4) submit();
    }});

    document.getElementById("undo").onclick = () => {{
      corners.pop();
      draw();
      updateStatus();
    }};
    document.getElementById("reset").onclick = () => {{
      corners = [];
      draw();
      updateStatus();
    }};
  </script>
</body>
</html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # silence request logs
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/done":
                qs = parse_qs(parsed.query)
                raw = qs.get("corners", [""])[0]
                pts: list[tuple[float, float]] = []
                for pair in raw.split(";"):
                    if not pair.strip():
                        continue
                    x_s, y_s = pair.split(",")
                    pts.append((float(x_s), float(y_s)))
                if len(pts) != 4:
                    self.send_response(400)
                    self.end_headers()
                    return
                result["corners"] = pts
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                ready.set()
                return
            self.send_response(404)
            self.end_headers()

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"
    print(f"Open {url} and click the 4 court corners in order:")
    for i, label in enumerate(CORNER_LABELS, 1):
        print(f"  {i}. {label}")
    webbrowser.open(url)

    ready.wait()
    server.shutdown()
    corners = result["corners"]
    assert corners is not None
    return corners


def main() -> None:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read {IMAGE_PATH}")
    corners = click_court_corners(image)
    print("corners (pixel xy):")
    for i, (x, y) in enumerate(corners, 1):
        print(f"  {i}: ({x:.1f}, {y:.1f})")
    print("as list:", corners)


if __name__ == "__main__":
    main()
