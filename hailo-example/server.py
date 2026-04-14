"""
Hailo Example App — Real-time Object Detection with Bounding Boxes

Flask web server providing:
  /              — Web UI with live camera + bounding box overlay
  /video_feed    — JPEG snapshot (polling-based, works through HA ingress)
  /api/status    — Health / device info
"""

import os
import json
import time
import logging
import threading

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory

# ── Configuration ────────────────────────────────────────────────────────────
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "/dev/video0")
PORT = int(os.environ.get("PORT", "8099"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("hailo_example")

app = Flask(__name__, static_folder="static")

# ── Globals ──────────────────────────────────────────────────────────────────
camera_lock = threading.Lock()
cap = None
current_frame = None
current_detections = []
camera_ok = False
hailo_ok = False
_rtsp_reader_running = False

# Hailo inference objects
vdevice = None
network_group = None
input_info = None
output_infos = None
infer_pipeline = None

# COCO class names for YOLOv6n
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Colors for bounding boxes (BGR)
BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255), (128, 0, 255),
]


# ── Hailo setup ──────────────────────────────────────────────────────────────
def init_hailo():
    global vdevice, network_group, input_info, output_infos, hailo_ok

    try:
        from hailo_platform import (
            VDevice, HEF, ConfigureParams,
            InputVStreamParams, OutputVStreamParams,
            HailoStreamInterface, HailoSchedulingAlgorithm,
        )
    except ImportError:
        logger.error("hailo_platform not found")
        return False

    try:
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.multi_process_service = True
        params.group_id = "SHARED"

        vdevice = VDevice(params)
        logger.info("Connected to HailoRT service")

        # Download model if needed
        model_dir = "/data/models"
        model_path = os.path.join(model_dir, "yolov6n.hef")
        arch = os.environ.get("HAILO_ARCH", "hailo8l")

        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(model_dir, exist_ok=True)
            url = f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/{arch}/yolov6n.hef"
            logger.info(f"Downloading model from {url}")
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Saved to {model_path}")

        hef = HEF(model_path)
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe
        )
        network_group = vdevice.configure(hef, configure_params)[0]

        input_info = hef.get_input_vstream_infos()[0]
        output_infos = hef.get_output_vstream_infos()

        logger.info(f"Model loaded: input={input_info.name} shape={input_info.shape}")
        for o in output_infos:
            logger.info(f"  output={o.name} shape={o.shape}")

        hailo_ok = True
        return True
    except Exception as e:
        logger.error(f"Hailo init failed: {e}")
        return False


def run_inference(frame):
    """Run YOLOv6n inference on a frame. Returns list of detections."""
    global network_group, input_info, output_infos

    if not hailo_ok or network_group is None:
        return []

    try:
        from hailo_platform import (
            InputVStreamParams, OutputVStreamParams, InferVStreams,
        )

        h, w, c = input_info.shape
        orig_h, orig_w = frame.shape[:2]

        # Preprocess: resize to model input
        resized = cv2.resize(frame, (w, h))

        input_vstreams_params = InputVStreamParams.make(network_group)
        output_vstreams_params = OutputVStreamParams.make(network_group)

        batch = np.expand_dims(resized, axis=0)

        with InferVStreams(
            network_group, input_vstreams_params, output_vstreams_params
        ) as pipeline:
            results = pipeline.infer({input_info.name: batch})

        # Parse NMS output — ragged array: list of 80 classes,
        # each class has shape (N_detections, 5) where 5 = [y1, x1, y2, x2, conf]
        # N_detections varies per class, so np.array() on the whole thing fails.
        detections = []
        for name, data in results.items():
            # data is a list: [batch][class_id] -> ndarray of shape (N, 5)
            # or list of lists for NMS output
            if isinstance(data, (list, tuple)):
                batch_data = data[0] if len(data) > 0 else data
            else:
                batch_data = data

            for cls_id, cls_dets in enumerate(batch_data):
                cls_arr = np.array(cls_dets)
                if cls_arr.ndim == 0 or cls_arr.size == 0:
                    continue
                if cls_arr.ndim == 1:
                    cls_arr = cls_arr.reshape(1, -1)
                # Each row: [y1, x1, y2, x2, confidence]
                for det in cls_arr:
                    if len(det) < 5:
                        continue
                    y1, x1, y2, x2, conf = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    detections.append({
                        "class_id": cls_id,
                        "class_name": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}",
                        "confidence": conf,
                        "bbox": [
                            int(x1 * orig_w),
                            int(y1 * orig_h),
                            int(x2 * orig_w),
                            int(y2 * orig_h),
                        ]
                    })

        return detections
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return []


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        color = BOX_COLORS[cls_id % len(BOX_COLORS)]
        label = f"{det['class_name']} {det['confidence']:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


# ── Camera helpers ───────────────────────────────────────────────────────────
def _is_network_url(source):
    return source.startswith(("rtsp://", "rtsps://", "http://", "https://"))


def _rtsp_reader_loop():
    global current_frame, camera_ok, cap, _rtsp_reader_running
    logger.info("RTSP reader thread started")
    consecutive_failures = 0
    while _rtsp_reader_running:
        if cap is None or not cap.isOpened():
            logger.warning("RTSP stream lost — reconnecting in 2s")
            time.sleep(2)
            try:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened():
                    camera_ok = True
                    consecutive_failures = 0
                    logger.info("RTSP stream reconnected")
            except Exception as e:
                logger.error(f"RTSP reconnect error: {e}")
            continue

        ret, frame = cap.read()
        if ret:
            current_frame = frame
            camera_ok = True
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures > 50:
                logger.warning("RTSP: too many failures, forcing reconnect")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                consecutive_failures = 0
            time.sleep(0.01)
            continue
        time.sleep(0.03)


def open_camera():
    global cap, camera_ok, _rtsp_reader_running
    try:
        src = CAMERA_SOURCE
        if _is_network_url(src):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            logger.info(f"Opening network camera (TCP): {src}")
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera_ok = cap.isOpened()
            if camera_ok:
                logger.info(f"Camera opened: {src}")
                _rtsp_reader_running = True
                t = threading.Thread(target=_rtsp_reader_loop, daemon=True)
                t.start()
            else:
                logger.warning(f"Camera could not be opened: {src}")
        else:
            if src.startswith("/dev/video"):
                idx = int(src.replace("/dev/video", ""))
            else:
                idx = int(src) if src.isdigit() else 0
            logger.info(f"Opening USB camera: {src} (index {idx})")
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            camera_ok = cap.isOpened()
            if camera_ok:
                logger.info(f"Camera opened: {src}")
            else:
                logger.warning(f"Camera could not be opened: {src}")
    except Exception as e:
        logger.error(f"Camera error: {e}")
        camera_ok = False


def read_frame():
    global current_frame
    if not camera_ok or cap is None:
        return current_frame
    if _rtsp_reader_running:
        return current_frame
    with camera_lock:
        ret, frame = cap.read()
    if ret:
        current_frame = frame
        return frame
    return current_frame


def generate_placeholder():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "No Camera", (180, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
    cv2.putText(img, f"Looking for {CAMERA_SOURCE}", (120, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    return img


# ── Inference loop ───────────────────────────────────────────────────────────
inference_lock = threading.Lock()
last_inference_time = 0
INFERENCE_INTERVAL = 0.15  # ~6-7 fps inference


def inference_loop():
    """Background thread: runs inference on latest frame."""
    global current_detections, last_inference_time
    while True:
        frame = current_frame
        if frame is not None and hailo_ok:
            with inference_lock:
                current_detections = run_inference(frame)
                last_inference_time = time.time()
        time.sleep(INFERENCE_INTERVAL)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/video_feed")
def video_feed():
    frame = read_frame()
    if frame is None:
        frame = generate_placeholder()
    else:
        frame = frame.copy()

    # Draw bounding boxes
    dets = current_detections
    if dets:
        frame = draw_detections(frame, dets)

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpeg.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store"})


@app.route("/api/status")
def api_status():
    return jsonify({
        "camera": camera_ok,
        "hailo": hailo_ok,
        "camera_source": CAMERA_SOURCE,
        "detections": len(current_detections),
        "last_inference": last_inference_time,
    })


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Hailo Example App — Real-time Detection")
    logger.info("=" * 60)

    open_camera()
    init_hailo()

    # Start inference thread
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    logger.info(f"Inference thread started (interval={INFERENCE_INTERVAL}s)")

    logger.info(f"Starting web server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
