from flask import Flask, render_template, Response
from flask_socketio import SocketIO

import cv2
import mediapipe as mp
import numpy as np
import joblib
import requests
import time
import os

from collections import deque, Counter

# ================= FLASK =================
app = Flask(__name__)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet"
)

# ================= MODEL =================
model = joblib.load("models/sibi_model.pkl")

# ================= ESP32 =================
ESP32_LED_IP = "192.168.137.68"
ESP32_CAM_IP = "192.168.137.55"

STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"

session = requests.Session()

# ================= LED =================
def send_led(status):

    try:

        session.get(
            f"http://{ESP32_LED_IP}/led?status={status}",
            timeout=0.3
        )

    except:
        pass

# ================= BUTTON =================
def get_button():

    try:

        r = session.get(
            f"http://{ESP32_LED_IP}/button",
            timeout=0.3
        )

        return r.text.strip()

    except:
        return "none"

# ================= CAMERA =================
def connect_camera():

    print("📷 Connecting ESP32-CAM...")

    cap = cv2.VideoCapture(STREAM_URL)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap

cap = connect_camera()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.75,

    min_tracking_confidence=0.75

)

# ================= STABILIZATION =================
history = deque(maxlen=25)

text_output = ""
current_word = ""

last_pred = ""
stable_start = 0

delay_letter = 2.0

last_led = -1
last_btn_time = 0

# ================= CLEAN =================
def clean_letter(letter):

    if len(letter) == 0:
        return ""

    return letter[0].upper()

# ================= GENERATE FRAMES =================
def generate_frames():

    global cap
    global text_output
    global current_word
    global last_pred
    global stable_start
    global last_led
    global last_btn_time

    while True:

        # ================= READ CAMERA =================
        ret, frame = cap.read()

        if not ret:

            print("🔄 Reconnect ESP32-CAM...")

            try:
                cap.release()
            except:
                pass

            time.sleep(1)

            cap = connect_camera()

            continue

        # ================= FRAME SIZE =================
        frame = cv2.resize(frame, (900, 520))

        # ================= RGB =================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        current_pred = ""

        # ================= HAND DETECTION =================
        if result.multi_hand_landmarks:

            # LED ON
            if last_led != 1:

                send_led(1)

                last_led = 1

            for hand_landmarks in result.multi_hand_landmarks:

                data = []

                for lm in hand_landmarks.landmark:

                    data.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                try:

                    pred = model.predict(
                        np.array([data])
                    )[0]

                    pred = clean_letter(str(pred))

                    history.append(pred)

                    current_pred = Counter(
                        history
                    ).most_common(1)[0][0]

                except:

                    current_pred = ""

                # ================= DRAW LANDMARK =================
                mp_draw.draw_landmarks(

                    frame,

                    hand_landmarks,

                    mp_hands.HAND_CONNECTIONS,

                    mp_draw.DrawingSpec(
                        color=(0,255,255),
                        thickness=2,
                        circle_radius=3
                    ),

                    mp_draw.DrawingSpec(
                        color=(255,0,255),
                        thickness=2
                    )
                )

        else:

            # LED OFF
            if last_led != 0:

                send_led(0)

                last_led = 0

        # ================= STABLE LETTER =================
        if current_pred:

            if current_pred != last_pred:

                last_pred = current_pred

                stable_start = time.time()

            else:

                duration = time.time() - stable_start

                if duration > delay_letter:

                    if (
                        len(current_word) == 0
                        or current_word[-1] != current_pred
                    ):

                        current_word += current_pred

                        print("✅ Huruf masuk:", current_pred)

                    last_pred = ""

        # ================= AUTO TEXT =================
        else:

            if current_word != "":

                text_output += current_word + " "

                print("📝 Kata masuk:", current_word)

                current_word = ""

        # ================= BUTTON =================
        if time.time() - last_btn_time > 0.4:

            btn = get_button()

            last_btn_time = time.time()

            if btn == "hapus":

                if len(current_word) > 0:

                    current_word = current_word[:-1]

                elif len(text_output) > 0:

                    text_output = text_output[:-1]

        # ================= SEND TO WEBSITE =================
        socketio.emit("prediction", {

            "huruf": current_pred,
            "kata": current_word,
            "teks": text_output

        })

        # ================= JPEG =================
        ret, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()

        # ================= STREAM =================
        yield (

            b'--frame\r\n'

            b'Content-Type: image/jpeg\r\n\r\n' +

            frame_bytes +

            b'\r\n'
        )

        # ================= CPU DELAY =================
        time.sleep(0.01)

# ================= ROUTES =================
@app.route("/")
def index():

    return render_template("index.html")

@app.route("/video")
def video():

    return Response(

        generate_frames(),

        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ================= RUN =================
if __name__ == "__main__":

    PORT = int(os.environ.get("PORT", 5000))

    socketio.run(

        app,

        host="0.0.0.0",

        port=PORT,

        debug=False
    )