import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

MODEL_PATH = 'hand_landmarker.task'

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

fist_start_time = None
FIST_CLEAR_DURATION = 3.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_hand_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for i, point in enumerate(points):
        color = (0, 0, 255) if i in [4, 8, 12, 16, 20] else (0, 255, 0)
        cv2.circle(image, point, 4, color, -1)

def is_finger_extended(hand_landmarks, tip_idx, mcp_idx):
    tip_y = hand_landmarks[tip_idx].y
    mcp_y = hand_landmarks[mcp_idx].y
    return tip_y < mcp_y

def is_fist(hand_landmarks):
    thumb_extended = is_finger_extended(hand_landmarks, 4, 1)
    index_extended = is_finger_extended(hand_landmarks, 8, 5)
    middle_extended = is_finger_extended(hand_landmarks, 12, 9)
    ring_extended = is_finger_extended(hand_landmarks, 16, 13)
    pinky_extended = is_finger_extended(hand_landmarks, 20, 17)
    return not (index_extended or middle_extended or ring_extended or pinky_extended)

def process_frame(image):
    global fist_start_time
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = hand_landmarker.detect(mp_image)

    fingertip = None
    index_open = False
    should_clear = False

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            index_open = is_finger_extended(hand_landmarks, 8, 5)
            other_extended = (
                is_finger_extended(hand_landmarks, 12, 9) or
                is_finger_extended(hand_landmarks, 16, 13) or
                is_finger_extended(hand_landmarks, 20, 17)
            )

            draw_hand_landmarks(image, hand_landmarks)
            h, w, _ = image.shape
            cx = int(hand_landmarks[8].x * w)
            cy = int(hand_landmarks[8].y * h)

            if is_fist(hand_landmarks):
                if fist_start_time is None:
                    fist_start_time = time.time()
                elif time.time() - fist_start_time >= FIST_CLEAR_DURATION:
                    should_clear = True
                    fist_start_time = None
            else:
                fist_start_time = None

            if index_open and not other_extended:
                fingertip = (cx, cy)
                cv2.circle(image, (cx, cy), 10, (0, 0, 0), 3)

    return image, fingertip, should_clear

@socketio.on('frame')
def handle_frame(data):
    encoded = data.split(',')[1] if ',' in data else data
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return

    annotated_frame, fingertip, should_clear = process_frame(frame)

    h, w = frame.shape[:2]
    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buffer).decode('utf-8')
    emit('processed_frame', {
        'image': b64,
        'fingertip': fingertip,
        'width': w,
        'height': h,
        'should_clear': should_clear
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=8080)