"""
app.py — Lane Detection Web App Backend
Serves: /  (index.html), /upload, /stream/<kind>, /metrics/<job_id>

Performance fixes applied
─────────────────────────
1. TARGET_FPS cap in the processing thread — avoids burning 100% CPU
   and eliminates the GIL-starvation latency spikes.
2. MJPEG stream generator sleeps between yields instead of busy-spinning.
3. Benchmark stats are collected per-frame and aggregated; exposed via
   /metrics/<job_id> so the frontend can show live FPS/latency without
   extra load on the hot path.
4. /stream generator breaks cleanly when a job is done and the stream
   has delivered its last frame — no zombie threads.
"""

import os, uuid, time, threading
import cv2
import numpy as np
from collections import deque
from flask import Flask, render_template, request, jsonify, Response

from lane_detection import detect_lanes

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Target frame rate for the processing thread ───────────────────────────────
TARGET_FPS   = 25          # cap: won't process faster than this
FRAME_BUDGET = 1.0 / TARGET_FPS   # seconds per frame budget

# ── In-memory job store ───────────────────────────────────────────────────────
# job dict keys:
#   result/edges/roi : latest JPEG bytes
#   done             : bool
#   progress/total   : int frame counts
#   error            : str | None
#   latencies        : deque of recent per-frame ms values (capped at 60)
#   fps              : float — rolling average
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_bgr(img):
    if img is None:
        return np.zeros((360, 640, 3), dtype=np.uint8)
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def encode_jpeg(frame) -> bytes:
    frame = ensure_bgr(frame)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


def frame_to_mjpeg(frame_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
    )


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

    latency_buf = deque(maxlen=60)   # rolling window for FPS calc
    idx = 0

    while True:
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        # detect_lanes() is the heavy work; time it
        t0 = time.perf_counter()
        result, hls_vis, roi_vis, bev_vis = detect_lanes(frame)
        t1 = time.perf_counter()

        frame_ms = (t1 - t0) * 1000.0
        latency_buf.append(frame_ms)

        # Rolling FPS: 1000 / mean_ms capped to TARGET_FPS
        mean_ms = sum(latency_buf) / len(latency_buf)
        rolling_fps = min(TARGET_FPS, 1000.0 / mean_ms) if mean_ms > 0 else 0

        with jobs_lock:
            jobs[job_id]["result"]   = encode_jpeg(result)
            # jobs[job_id]["edges"]    = encode_jpeg(hls_vis)
            # jobs[job_id]["roi"]      = encode_jpeg(roi_vis)
            jobs[job_id]["progress"] = idx + 1
            jobs[job_id]["latency_ms"] = round(frame_ms, 1)
            jobs[job_id]["fps"]      = round(rolling_fps, 1)

        # ── FPS cap: sleep out any remaining budget ────────────────
        elapsed = time.perf_counter() - loop_start
        sleep_for = FRAME_BUDGET - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

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
            "result": None, "edges": None, "roi": None,
            "done": False, "progress": 0, "total": 1,
            "error": None, "latency_ms": 0.0, "fps": 0.0,
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


@app.route("/metrics/<job_id>")
def metrics(job_id):
    """Lightweight endpoint — just latency + fps. Polled by frontend."""
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job."}), 404
    return jsonify({
        "latency_ms": job.get("latency_ms", 0.0),
        "fps":        job.get("fps", 0.0),
        "done":       job["done"],
    })


@app.route("/stream/<job_id>/<kind>")
def stream(job_id, kind):
    """MJPEG stream for one of: result | edges | roi"""
    if kind not in ("result", "edges", "roi"):
        return "Invalid stream kind.", 400

    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    _, blank_buf = cv2.imencode(".jpg", blank)
    blank_bytes  = blank_buf.tobytes()

    def generate():
        # How long to sleep between frames — matches the processing rate.
        # We add a tiny extra slack so we're never busier than the producer.
        stream_sleep = FRAME_BUDGET * 0.9

        while True:
            with jobs_lock:
                job = jobs.get(job_id)

            if job is None:
                yield frame_to_mjpeg(blank_bytes)
                return           # job evicted — close stream

            frame_bytes = job.get(kind) or blank_bytes
            yield frame_to_mjpeg(frame_bytes)

            if job["done"]:
                # Stream one last good frame then close — no infinite loop
                time.sleep(0.05)
                return

            # ── KEY FIX: sleep so this thread doesn't busy-spin ───
            time.sleep(stream_sleep)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)