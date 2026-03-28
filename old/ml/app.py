"""
app.py — Lane Detection Web App Backend
Serves: /  (index.html), /upload, /status/<id>, /stream/<id>/<kind>

Stream kinds
────────────
  result   — final lane-overlay frame
  edges    — raw Canny edge image
  warped   — bird's-eye binary (NEW — replaces old "roi" stream)
  windows  — sliding-window debug view (NEW)

Change from old version
───────────────────────
  detect_lanes() now returns 4 values:
      result, warped_bin, edges, debug_win
  The old "roi" key is replaced by "warped" for clarity.
  A new "windows" key carries the sliding-window debug frame.
  The /stream route accepts both old name "roi" (redirects → warped)
  and all new names so existing front-ends don't break immediately.
"""

import os
import time
import uuid
import threading

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response

from lane_detection import detect_lanes

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────────────
# job_id → {
#   "result":  bytes | None,
#   "edges":   bytes | None,
#   "warped":  bytes | None,   ← bird's-eye binary (was "roi")
#   "windows": bytes | None,   ← sliding-window debug (NEW)
#   "done":    bool,
#   "progress": int,
#   "total":   int,
#   "error":   str | None,
# }
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

STREAM_KINDS = {"result", "edges", "warped", "windows"}
# back-compat alias: old front-ends that request "roi" get "warped"
_KIND_ALIAS  = {"roi": "warped"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_jpeg(frame) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def frame_to_mjpeg(frame_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
    )


def _gray_to_bgr(frame):
    """Convert grayscale to BGR so all streams are consistent colour."""
    if frame is not None and len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


# ── Background processing thread ──────────────────────────────────────────────

def process_video_job(job_id: str, video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with jobs_lock:
            jobs[job_id]["error"] = "Cannot open video."
            jobs[job_id]["done"]  = True
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    with jobs_lock:
        jobs[job_id]["total"] = total

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect_lanes returns 4 values now
        result, warped_bin, edges, debug_win = detect_lanes(frame, show_debug=True)

        # Convert any grayscale debug frames to BGR
        edges_bgr   = _gray_to_bgr(edges)
        warped_bgr  = _gray_to_bgr(warped_bin)
        # debug_win is already BGR (or None)
        windows_bgr = debug_win if debug_win is not None else warped_bgr

        with jobs_lock:
            jobs[job_id]["result"]   = encode_jpeg(result)
            jobs[job_id]["edges"]    = encode_jpeg(edges_bgr)
            jobs[job_id]["warped"]   = encode_jpeg(warped_bgr)
            jobs[job_id]["windows"]  = encode_jpeg(windows_bgr)
            jobs[job_id]["progress"] = idx + 1

        idx += 1

    cap.release()
    try:
        os.remove(video_path)
    except OSError:
        pass

    with jobs_lock:
        jobs[job_id]["done"] = True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    job_id = str(uuid.uuid4())
    ext    = os.path.splitext(f.filename)[1].lower() or ".mp4"
    path   = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    f.save(path)

    with jobs_lock:
        jobs[job_id] = {
            "result": None, "edges": None,
            "warped": None, "windows": None,
            "done": False, "progress": 0, "total": 1, "error": None,
        }

    t = threading.Thread(
        target=process_video_job, args=(job_id, path), daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job."}), 404
    return jsonify({
        "done":     job["done"],
        "progress": job["progress"],
        "total":    job["total"],
        "error":    job["error"],
    })


@app.route("/stream/<job_id>/<kind>")
def stream(job_id, kind):
    """
    MJPEG stream.  kind = result | edges | warped | windows
    Legacy alias: roi → warped
    """
    kind = _KIND_ALIAS.get(kind, kind)          # resolve alias
    if kind not in STREAM_KINDS:
        return "Invalid stream kind. Use: result, edges, warped, windows", 400

    # Reusable blank placeholder frame
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", blank)
    blank_bytes = buf.tobytes()

    def generate():
        while True:
            with jobs_lock:
                job = jobs.get(job_id)

            if job is None:
                yield frame_to_mjpeg(blank_bytes)
                break

            frame_bytes = job.get(kind) or blank_bytes
            yield frame_to_mjpeg(frame_bytes)

            if job["done"]:
                # Keep last frame alive; client can disconnect
                time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)