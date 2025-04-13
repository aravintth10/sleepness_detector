import cv2
import mediapipe as mp
import time
from playsound import playsound
import threading
from datetime import datetime

# Constants
EYE_AR_THRESH = 0.25
SLEEPY_DURATION_THRESHOLD = 6  # seconds
LOG_FILE = "sleepy_log.txt"

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def play_alert():
    playsound("alert.mp3")  # Replace with your actual alarm sound path

def log_sleepy_event(duration):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Sleepy for {duration:.1f} seconds\n")

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    def _p(index):
        return int(landmarks[index].x * image_w), int(landmarks[index].y * image_h)

    p1, p2, p3, p4, p5, p6 = [_p(i) for i in eye_indices]

    A = ((p2[0] - p6[0]) ** 2 + (p2[1] - p6[1]) ** 2) ** 0.5
    B = ((p3[0] - p5[0]) ** 2 + (p3[1] - p5[1]) ** 2) ** 0.5
    C = ((p1[0] - p4[0]) ** 2 + (p1[1] - p4[1]) ** 2) ** 0.5

    ear = (A + B) / (2.0 * C)
    return ear

# Initialize
cap = cv2.VideoCapture(0)
eyes_closed_start = None
alert_triggered = False
sleepy_counter = 0
last_sleepy_time = ""
detection_active = False  # To track start/stop state

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    h, w, _ = frame.shape

    # Check for start/stop
    cv2.putText(frame, "Press 'S' to Start/Resume, 'P' to Pause, 'Q' to Quit", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Start or resume
        detection_active = True
    elif key == ord('p'):  # Pause
        detection_active = False
    elif key == ord('q'):  # Quit
        break

    if detection_active:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()

                    elapsed = time.time() - eyes_closed_start

                    # Progress bar
                    cv2.rectangle(frame, (30, 130), (30 + int((elapsed / SLEEPY_DURATION_THRESHOLD) * 200), 150), (0, 0, 255), -1)
                    cv2.rectangle(frame, (30, 130), (230, 150), (255, 255, 255), 2)

                    if elapsed >= SLEEPY_DURATION_THRESHOLD and not alert_triggered:
                        threading.Thread(target=play_alert, daemon=True).start()
                        alert_triggered = True
                        sleepy_counter += 1
                        last_sleepy_time = datetime.now().strftime('%H:%M:%S')
                        log_sleepy_event(elapsed)

                    cv2.putText(frame, f"SLEEPY for {int(elapsed)}s", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    eyes_closed_start = None
                    alert_triggered = False

        # Display sleepy counter and last sleepy time
        cv2.putText(frame, f"Sleepy Count: {sleepy_counter}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if last_sleepy_time:
            cv2.putText(frame, f"Last Sleepy: {last_sleepy_time}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Detection Paused", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sleep Detection", frame)

cap.release()
cv2.destroyAllWindows()
