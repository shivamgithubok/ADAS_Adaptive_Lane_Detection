"""
app.py — Lane Detection Web App Backend
Serves: /  (index.html), /upload, /stream/<kind>
"""

import os
import uuid
import threading
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response

from lane_detection import detect_lanes

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── In-memory store of processed frame buffers ────────────────────────────────
# Each job_id → {"result": bytes, "edges": bytes, "roi": bytes, "done": bool,
#                "progress": int, "total": int, "error": str|None}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_jpeg(frame) -> bytes:
    """Encode an OpenCV frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
            jobs[job_id]["done"] = True
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    with jobs_lock:
        jobs[job_id]["total"] = total

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, roi, edges, _ = detect_lanes(frame)

        # Convert grayscale debug frames to BGR so they look consistent
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        roi_bgr   = cv2.cvtColor(roi,   cv2.COLOR_GRAY2BGR)

        with jobs_lock:
            jobs[job_id]["result"] = encode_jpeg(result)
            jobs[job_id]["edges"]  = encode_jpeg(edges_bgr)
            jobs[job_id]["roi"]    = encode_jpeg(roi_bgr)
            jobs[job_id]["progress"] = idx + 1

        idx += 1

    cap.release()
    os.remove(video_path)          # clean up upload

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
    ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
    path = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    f.save(path)

    with jobs_lock:
        jobs[job_id] = {
            "result": None, "edges": None, "roi": None,
            "done": False, "progress": 0, "total": 1, "error": None,
        }

    t = threading.Thread(target=process_video_job, args=(job_id, path), daemon=True)
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
    """MJPEG stream for one of: result | edges | roi"""
    if kind not in ("result", "edges", "roi"):
        return "Invalid stream kind.", 400

    # Placeholder black frame
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    _, blank_buf = cv2.imencode(".jpg", blank)
    blank_bytes = blank_buf.tobytes()

    def generate():
        while True:
            with jobs_lock:
                job = jobs.get(job_id)

            if job is None:
                yield frame_to_mjpeg(blank_bytes)
                break

            frame_bytes = job.get(kind) or blank_bytes
            yield frame_to_mjpeg(frame_bytes)

            if job["done"] and job.get(kind):
                # Keep streaming the last frame
                import time
                time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)