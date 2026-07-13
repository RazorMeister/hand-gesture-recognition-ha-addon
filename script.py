import argparse
import sys
import time
import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
import logging
import json
import os
from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ if present (standalone runs)
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
  global FRAME_COUNT, mqtt_last_restart_time # Declare FRAME_COUNT as a global variable
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

  recognition_frame = None
  recognition_result_list = []
  

  def save_result(result: vision.GestureRecognizerResult,
                  unused_output_image: mp.Image, timestamp_ms: int):
      global FPS, COUNTER, START_TIME, _LAST_DIAG

      # Calculate the FPS
      if COUNTER % fps_avg_frame_count == 0:
          FPS = fps_avg_frame_count / (time.time() - START_TIME)
          START_TIME = time.time()

      # Diagnostic: tells you WHERE the pipeline fails. Logged only on state
      # change so it doesn't spam every sampled frame.
      #   no landmarks  -> palm detector/landmark stage (camera/framing/light)
      #   landmarks but no gesture -> classifier problem (retraining would help)
      if not result.hand_landmarks:
          diag = "no_hand"
      elif not result.gestures:
          diag = "no_gesture"
      else:
          diag = "ok"
      if diag != _LAST_DIAG:
          if diag == "no_hand":
              logger.info("diag: no hand detected in frame")
          elif diag == "no_gesture":
              logger.info("diag: hand detected but no gesture classified")
          else:
              logger.info("diag: hand + gesture ok")
          _LAST_DIAG = diag

      recognition_result_list.append(result)
      COUNTER += 1

  # Initialize the gesture recognizer model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.GestureRecognizerOptions(base_options=base_options,
                                          running_mode=vision.RunningMode.LIVE_STREAM,
                                          num_hands=num_hands,
                                          min_hand_detection_confidence=min_hand_detection_confidence,
                                          min_hand_presence_confidence=min_hand_presence_confidence,
                                          min_tracking_confidence=min_tracking_confidence,
                                          result_callback=save_result)
  recognizer = vision.GestureRecognizer.create_from_options(options)

    
  # Per-hand state, keyed by hand_index (MediaPipe gives no person identity,
  # so hands are just slots 0..num_hands-1). Separate slots -> two hands don't
  # clobber each other's dedup state.
  prev_handedness_value = {}
  hand_time = {}

  # Continuously capture images from the camera and run inference
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if ENHANCE_CONTRAST else None
  last_analysis = 0.0  # wall-clock time of the last analysed frame
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
    if MOTION_THRESHOLD > 0:
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 90))
        motion = 999.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean())
        prev_gray = gray
        if motion < MOTION_THRESHOLD:
            continue

    # Optional contrast boost: apply CLAHE on the L (lightness) channel to pull
    # detail out of backlit/shadowed regions without wrecking colour.
    if clahe is not None:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Feed the frame straight from memory (no lossy frame.jpg round-trip).
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)


    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)

    if recognition_result_list:
      #print(recognition_result_list)
      # Draw landmarks and write the text for each hand.
      for hand_index, hand_landmarks in enumerate(
          recognition_result_list[0].hand_landmarks):
        # Calculate the bounding box of the hand
        x_min = min([landmark.x for landmark in hand_landmarks])
        y_min = min([landmark.y for landmark in hand_landmarks])
        y_max = max([landmark.y for landmark in hand_landmarks])

        # Convert normalized coordinates to pixel values
        frame_height, frame_width = current_frame.shape[:2]
        x_min_px = int(x_min * frame_width)
        y_min_px = int(y_min * frame_height)
        y_max_px = int(y_max * frame_height)

        #Get hand 
        if recognition_result_list[0].handedness:
           handedness_info = recognition_result_list[0].handedness[0]
           handedness_value = handedness_info[0].display_name
           #print(handedness_value)

        # Get gesture classification results
        if recognition_result_list[0].gestures:
          
          gesture = recognition_result_list[0].gestures[hand_index]
          category_name = gesture[0].category_name
          score = round(gesture[0].score, 2)
          result_text = f'{category_name} ({score})'
        
        

          # Compute text size
          text_size = \
          cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                          label_thickness)[0]
          text_width, text_height = text_size

          # Calculate text position (above the hand)
          text_x = x_min_px
          text_y = y_min_px - 10  # Adjust this value as needed

          # Make sure the text is within the frame boundaries
          if text_y < 0:
            text_y = y_max_px + text_height

          # Draw the text
          cv2.putText(current_frame, result_text, (text_x, text_y),
                      cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                      label_text_color, label_thickness, cv2.LINE_AA)
          
          #print(result_text, (text_x, text_y))
        hand_status = category_name

        # Option A: every detected hand publishes to one topic. Skip the
        # "None"/no-gesture class so idle hands don't spam MQTT.
        if hand_status in ("None", ""):
            continue

        # Per-hand dedup (keyed by hand_index). Reset this hand's memory after
        # reset_hand_status_time seconds so the same gesture can fire again.
        reset_after = int(data.get("reset_hand_status_time"))
        if hand_index in hand_time and \
           (time.time() - hand_time[hand_index]) >= reset_after:
              prev_handedness_value.pop(hand_index, None)

        # Publish only on change for this hand, above confidence threshold.
        if hand_status != prev_handedness_value.get(hand_index) and score > MIN_GESTURE_SCORE:
              client.publish(mqtt_topic, hand_status)
              logger.info("hand %d: %s (%.2f)", hand_index, hand_status, score)
              prev_handedness_value[hand_index] = hand_status
              hand_time[hand_index] = time.time()
              
        # Draw hand landmarks on the frame
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                          z=landmark.z) for landmark in
          hand_landmarks
        ])
        mp_drawing.draw_landmarks(
          current_frame,
          hand_landmarks_proto,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

      recognition_frame = current_frame
      recognition_result_list.clear()

    #if recognition_frame is not None:
       # cv2.imshow('gesture_recognition', recognition_frame)

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

  # Config options win over argparse defaults so HA users can tune sensitivity.
  num_hands = int(data.get("num_hands", args.numHands))
  det_conf = float(data.get("min_hand_detection_confidence", args.minHandDetectionConfidence))
  pres_conf = float(data.get("min_hand_presence_confidence", args.minHandPresenceConfidence))
  track_conf = float(data.get("min_tracking_confidence", args.minTrackingConfidence))
  run(args.model, num_hands, det_conf, pres_conf, track_conf,
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()
