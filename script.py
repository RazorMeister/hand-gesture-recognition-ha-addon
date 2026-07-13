import argparse
import sys
import time
import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
import logging
import json
import os
import threading
from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ if present (standalone runs)

try:
    from flask import Flask, Response
except ImportError:
    Flask = None  # web UI simply disabled if flask isn't installed
json_file_path = '/data/options.json'
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
if os.path.exists(json_file_path):
    with open(json_file_path, 'r') as file:
        json_data = file.read()
else:
    # /data/options.json only exists when run as an HA addon via Supervisor.
    # Fall back to environment variables for standalone/docker runs.
    json_data = json.dumps({
        "rtsp_url": os.environ.get("RTSP_URL", ""),
        "mqtt_host": os.environ.get("MQTT_HOST", ""),
        "mqtt_port": int(os.environ.get("MQTT_PORT", "1883")),
        "mqtt_username": os.environ.get("MQTT_USERNAME", ""),
        "mqtt_password": os.environ.get("MQTT_PASSWORD", ""),
        "mqtt_topic": os.environ.get("MQTT_TOPIC", "hand_gesture_status"),
        "mqtt_enable_topic": os.environ.get("MQTT_ENABLE_TOPIC", "hand_gesture_enable"),
        "reset_hand_status_time": int(os.environ.get("RESET_HAND_STATUS_TIME", "10")),
        "roi_top": float(os.environ.get("ROI_TOP", "0.0")),
        "roi_bottom": float(os.environ.get("ROI_BOTTOM", "1.0")),
        "roi_left": float(os.environ.get("ROI_LEFT", "0.0")),
        "roi_right": float(os.environ.get("ROI_RIGHT", "1.0")),
        "num_hands": int(os.environ.get("NUM_HANDS", "4")),
        "min_hand_detection_confidence": float(os.environ.get("MIN_HAND_DETECTION_CONFIDENCE", "0.3")),
        "min_hand_presence_confidence": float(os.environ.get("MIN_HAND_PRESENCE_CONFIDENCE", "0.5")),
        "min_tracking_confidence": float(os.environ.get("MIN_TRACKING_CONFIDENCE", "0.5")),
        "min_gesture_score": float(os.environ.get("MIN_GESTURE_SCORE", "0.5")),
        "analyze_interval": float(os.environ.get("ANALYZE_INTERVAL", "0.4")),
        "enhance_contrast": os.environ.get("ENHANCE_CONTRAST", "false"),
        "motion_threshold": float(os.environ.get("MOTION_THRESHOLD", "3.0")),
        "web_ui": os.environ.get("WEB_UI", "true"),
        "web_port": int(os.environ.get("WEB_PORT", "8099")),
        "zones": int(os.environ.get("ZONES", "1")),
    })


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logging level to INFO or any other level you prefer

# Define a handler to output logs to standard output
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set the level for this handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parse the JSON data
data = json.loads(json_data)



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MQTT configuration
mqtt_broker_address = data.get("mqtt_host")
mqtt_port = data.get("mqtt_port")
mqtt_topic = data.get("mqtt_topic")
mqtt_username = data.get("mqtt_username")
mqtt_password = data.get("mqtt_password")
# Topic HA publishes ON/OFF to, mirroring input_boolean.cfg_camera_gesture_recognition.
mqtt_enable_topic = data.get("mqtt_enable_topic", "hand_gesture_enable")

# Optional region-of-interest crop, as fractions of the frame (0..1). Restrict
# where a hand is searched for -> hand is larger after MediaPipe's internal
# resize -> better detection. Default = full frame (no crop).
ROI_TOP = float(data.get("roi_top", 0.0))
ROI_BOTTOM = float(data.get("roi_bottom", 1.0))
ROI_LEFT = float(data.get("roi_left", 0.0))
ROI_RIGHT = float(data.get("roi_right", 1.0))
ROI_ENABLED = (ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT) != (0.0, 1.0, 0.0, 1.0)

# Minimum gesture-classification score to publish. Lower = more sensitive
# (fires on weaker gestures, but more false positives).
MIN_GESTURE_SCORE = float(data.get("min_gesture_score", 0.5))

# Optional CLAHE contrast boost. Helps backlit / uneven-light scenes (e.g. a
# bright window on one side) where a hand in shadow has too little contrast for
# the palm detector. Costs a little CPU per analysed frame.
ENHANCE_CONTRAST = str(data.get("enhance_contrast", False)).lower() in ("true", "1", "yes", "on")

# Motion gate: skip the (expensive) MediaPipe pipeline when the ROI is basically
# static - nobody moving on the couch. Value = mean abs frame difference (0..255)
# on a tiny greyscale image. 0 disables the gate (always analyse).
MOTION_THRESHOLD = float(data.get("motion_threshold", 3.0))

# Web UI: serve an MJPEG preview of what the detector sees (ROI crop with hand
# landmarks + gesture labels + status). With host_network the port is reachable
# directly at http://<pi-ip>:WEB_PORT.
WEB_UI = str(data.get("web_ui", True)).lower() in ("true", "1", "yes", "on")
WEB_PORT = int(data.get("web_port", 8099))

# Split the ROI into this many vertical columns, each detected separately. A
# near-square zone lets MediaPipe's letterbox shrink the hand less -> better
# detection of small/distant hands. Zones also map to people sitting side by
# side (zone 0 = leftmost); each zone publishes to its own MQTT topic.
ZONES = max(1, int(data.get("zones", 1)))
_web_lock = threading.Lock()
_web_jpeg = None  # latest annotated JPEG bytes for the web UI


def set_web_frame(frame_bgr, status_lines):
    """Overlay status text, JPEG-encode, and stash for the web stream."""
    global _web_jpeg
    if not WEB_UI or Flask is None or frame_bgr is None:
        return
    disp = frame_bgr.copy()
    y = 22
    for line in status_lines:
        # Black outline + green text so it's readable on any background.
        cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24
    ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _web_lock:
            _web_jpeg = buf.tobytes()


def start_web_server():
    """Start the MJPEG preview server in a daemon thread."""
    if not WEB_UI or Flask is None:
        logger.info("Web UI disabled (web_ui=%s, flask=%s)", WEB_UI, Flask is not None)
        return
    app = Flask(__name__)

    @app.route("/")
    def _index():
        # Relative img src so it also works behind a reverse proxy.
        return ('<!doctype html><title>Gesture detector</title>'
                '<body style="margin:0;background:#111;text-align:center">'
                '<img src="stream" style="max-width:100%;height:auto"></body>')

    @app.route("/stream")
    def _stream():
        def gen():
            while True:
                with _web_lock:
                    frame = _web_jpeg
                if frame is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.1)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=WEB_PORT, threaded=True),
        daemon=True,
    ).start()
    logger.info("Web UI on http://<host>:%d", WEB_PORT)

# When False, the loop skips all heavy recognition work (palm detect + landmarks
# + classify) but keeps the RTSP stream and MQTT connection alive. Defaults True
# so the addon still works if HA never publishes an enable state.
analysis_enabled = True


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT Broker")
        client.subscribe(mqtt_enable_topic)
        logger.info("Subscribed to enable topic: %s", mqtt_enable_topic)
    else:
        logger.info("Connection to MQTT Broker failed with code %s", rc)

def on_message(client, userdata, msg):
    global analysis_enabled
    if msg.topic != mqtt_enable_topic:
        return
    payload = msg.payload.decode(errors="ignore").strip().lower()
    analysis_enabled = payload in ("on", "1", "true", "enabled", "yes")
    logger.info("Analysis %s (via %s=%s)",
                "ENABLED" if analysis_enabled else "DISABLED",
                mqtt_enable_topic, payload)

# Initialize MQTT client
client = mqtt.Client()

# Set credentials for broker
client.username_pw_set(username=mqtt_username, password=mqtt_password)

# Assign callback functions
client.on_connect = on_connect
client.on_message = on_message


# Connect to broker
client.connect(mqtt_broker_address, mqtt_port, 60)

# Start the loop
client.loop_start()
        



# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
FRAME_COUNT = 0  # Counter for saving images
ANALYZE_INTERVAL = float(data.get("analyze_interval", 0.4))  # seconds between analyses (lower = faster reaction, more CPU)
_LAST_DIAG = None  # last diagnostic state, so we log only on change (no spam)

def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
  global FRAME_COUNT, mqtt_last_restart_time, _LAST_DIAG, FPS
  """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the gesture recognition model bundle.
      num_hands: Max number of hands can be detected by the recognizer.
      min_hand_detection_confidence: The minimum confidence score for hand
        detection to be considered successful.
      min_hand_presence_confidence: The minimum confidence score of hand
        presence score in the hand landmark detection.
      min_tracking_confidence: The minimum confidence score for the hand
        tracking to be considered successful.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

  # Start capturing video input from the camera
  #cap = cv2.VideoCapture(camera_id)

  cap = cv2.VideoCapture(data.get("rtsp_url"))
 # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
 # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
 # cv2.namedWindow('gesture_recognition', cv2.WINDOW_NORMAL)
 # cv2.resizeWindow('gesture_recognition', 640, 480)
  
 

  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 0)  # black
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Label box parameters
  label_text_color = (255, 255, 255)  # white
  label_font_size = 1
  label_thickness = 2

  # IMAGE mode: synchronous recognize() returns the result directly, so we can
  # run detection on several zone crops per cycle and know which zone each hit
  # belongs to (LIVE_STREAM's async callback can't tell them apart).
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.GestureRecognizerOptions(base_options=base_options,
                                          running_mode=vision.RunningMode.IMAGE,
                                          num_hands=num_hands,
                                          min_hand_detection_confidence=min_hand_detection_confidence,
                                          min_hand_presence_confidence=min_hand_presence_confidence,
                                          min_tracking_confidence=min_tracking_confidence)
  recognizer = vision.GestureRecognizer.create_from_options(options)

    
  # Per-hand state, keyed by hand_index (MediaPipe gives no person identity,
  # so hands are just slots 0..num_hands-1). Separate slots -> two hands don't
  # clobber each other's dedup state.
  prev_handedness_value = {}
  hand_time = {}

  # Continuously capture images from the camera and run inference
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if ENHANCE_CONTRAST else None
  last_analysis = 0.0  # wall-clock time of the last analysed frame
  prev_cycle_t = 0.0   # for the analyse-rate FPS shown in the preview
  prev_gray = None     # previous tiny greyscale frame, for the motion gate
  while cap.isOpened():
    # HA toggled recognition off -> skip decode + heavy work, keep stream alive.
    if not analysis_enabled:
        cap.grab()          # advance stream without decoding
        time.sleep(0.05)    # ease CPU while idle
        continue

    # Time-based sampling: analyse at most once per ANALYZE_INTERVAL seconds,
    # independent of camera fps. grab() advances the RTSP stream cheaply
    # (no H264 decode) between analysed frames -> big CPU saving.
    now = time.time()
    if now - last_analysis < ANALYZE_INTERVAL:
        cap.grab()
        continue
    last_analysis = now

    success, image = cap.read()   # grab + decode the frame we analyse
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    # Optional ROI crop: hand only appears in part of the frame. Cropping makes
    # the hand larger after MediaPipe's internal resize -> better detection.
    if ROI_ENABLED:
        h, w = image.shape[:2]
        image = image[int(ROI_TOP * h):int(ROI_BOTTOM * h),
                      int(ROI_LEFT * w):int(ROI_RIGHT * w)]

    # Motion gate: on a static scene, skip the whole MediaPipe pipeline. Cheap
    # mean-abs-diff on a tiny greyscale frame; a hand moving easily clears it.
    motion = 0.0
    if MOTION_THRESHOLD > 0:
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 90))
        motion = 999.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean())
        prev_gray = gray
        if motion < MOTION_THRESHOLD:
            set_web_frame(image, ["IDLE - no motion (%.1f < %.1f)" % (motion, MOTION_THRESHOLD),
                                  "ROI x %.2f-%.2f  y %.2f-%.2f" % (ROI_LEFT, ROI_RIGHT, ROI_TOP, ROI_BOTTOM)])
            continue

    # Optional contrast boost: apply CLAHE on the L (lightness) channel to pull
    # detail out of backlit/shadowed regions without wrecking colour.
    if clahe is not None:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Split the ROI into ZONES vertical columns and detect in each. A narrower,
    # near-square column lets MediaPipe's letterbox shrink the hand less -> small
    # / distant hands detect better than in one wide strip. zone 0 = leftmost.
    roi_h, roi_w = image.shape[:2]
    display = image.copy()
    zone_w = max(1, roi_w // ZONES)
    reset_after = int(data.get("reset_hand_status_time"))
    ts = time.time()
    any_hand = any_gesture = False

    for z in range(ZONES):
        zx0 = z * zone_w
        zx1 = roi_w if z == ZONES - 1 else (z + 1) * zone_w
        zw = zx1 - zx0
        zone_img = image[:, zx0:zx1]

        if ZONES > 1:  # draw the zone boundary on the preview
            cv2.rectangle(display, (zx0, 0), (zx1 - 1, roi_h - 1), (255, 255, 0), 1)

        rgb = cv2.cvtColor(zone_img, cv2.COLOR_BGR2RGB)
        result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            continue
        any_hand = True

        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            # Landmarks are normalized to the zone; remap x into the full ROI so
            # the skeleton draws in the right place on the preview.
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=(zx0 + lm.x * zw) / roi_w, y=lm.y, z=lm.z)
                for lm in hand_landmarks])
            mp_drawing.draw_landmarks(
                display, proto, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            if not result.gestures or hand_index >= len(result.gestures):
                continue
            category_name = result.gestures[hand_index][0].category_name
            score = round(result.gestures[hand_index][0].score, 2)
            any_gesture = True

            # Gesture label above the hand on the preview.
            lx = int(zx0 + min(lm.x for lm in hand_landmarks) * zw)
            ly = max(18, int(min(lm.y for lm in hand_landmarks) * roi_h) - 8)
            cv2.putText(display, "%s (%.2f)" % (category_name, score), (lx, ly),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Skip the "None"/no-gesture class.
            if category_name in ("None", ""):
                continue

            # Zones are only a detection trick - everything still goes to the
            # single mqtt_topic. Dedup keyed per (zone, hand) so zones don't
            # clobber each other's state.
            key = (z, hand_index)
            if key in hand_time and (ts - hand_time[key]) >= reset_after:
                prev_handedness_value.pop(key, None)
            if category_name != prev_handedness_value.get(key) and score > MIN_GESTURE_SCORE:
                client.publish(mqtt_topic, category_name)
                logger.info("zone %d hand %d: %s (%.2f)",
                            z, hand_index, category_name, score)
                prev_handedness_value[key] = category_name
                hand_time[key] = ts

    # Analyse-rate FPS for the status overlay.
    FPS = 1.0 / max(1e-3, ts - prev_cycle_t)
    prev_cycle_t = ts

    # Diagnostic (logged only on state change).
    diag = "ok" if any_gesture else ("no_gesture" if any_hand else "no_hand")
    if diag != _LAST_DIAG:
        logger.info("diag: %s", diag)
        _LAST_DIAG = diag

    set_web_frame(display, [
        "FPS %.1f  motion %.1f  diag %s  zones %d" % (FPS, motion, diag, ZONES),
        "ROI x %.2f-%.2f  y %.2f-%.2f" % (ROI_LEFT, ROI_RIGHT, ROI_TOP, ROI_BOTTOM),
    ])

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

  recognizer.close()
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of gesture recognition model.',
      required=False,
      default='gesture_recognizer.task')
  parser.add_argument(
      '--numHands',
      help='Max number of hands that can be detected by the recognizer.',
      required=False,
      default=1)
  parser.add_argument(
      '--minHandDetectionConfidence',
      help='The minimum confidence score for hand detection to be considered '
           'successful.',
      required=False,
      default=0.3)
  parser.add_argument(
      '--minHandPresenceConfidence',
      help='The minimum confidence score of hand presence score in the hand '
           'landmark detection.',
      required=False,
      default=0.5)
  parser.add_argument(
      '--minTrackingConfidence',
      help='The minimum confidence score for the hand tracking to be '
           'considered successful.',
      required=False,
      default=0.5)
  # Finding the camera ID can be very reliant on platform-dependent methods.
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  # Start the MJPEG preview server (no-op if disabled / flask missing).
  start_web_server()

  # Config options win over argparse defaults so HA users can tune sensitivity.
  num_hands = int(data.get("num_hands", args.numHands))
  det_conf = float(data.get("min_hand_detection_confidence", args.minHandDetectionConfidence))
  pres_conf = float(data.get("min_hand_presence_confidence", args.minHandPresenceConfidence))
  track_conf = float(data.get("min_tracking_confidence", args.minTrackingConfidence))
  run(args.model, num_hands, det_conf, pres_conf, track_conf,
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()
