import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from tkinter import *
from gtts import gTTS
from playsound import playsound
import os

sign_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Apple','Can','Get','Good','Give me a call','I love you','I want money','I want to go to the washroom','Please stop','Thank you very much']

STGCN_MODEL_PATH = "model/stgcn_model.h5"
YOLO_MODEL_PATH = "model/yolo11_best.pt"

stgcn_model = load_model(STGCN_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def texttospeech(text, filename="voice"):
    filename = filename + '.mp3'
    if os.path.exists(filename):
        os.remove(filename)
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

def extract_skeleton(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])
        return np.array(keypoints)
    return None

def predict_sign(keypoints):
    x = np.expand_dims(keypoints, axis=0)
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=-1)
    prediction = stgcn_model.predict(x, verbose=0)
    return sign_labels[np.argmax(prediction)]

def detect_hand(frame):
    detections = yolo_model(frame)[0]
    for data in detections.boxes.data.tolist():
        xmin, ymin, xmax, ymax = map(int, data[:4])
        return frame, (xmin, ymin, xmax, ymax)
    return frame, None

def signfromWebcam():
    camera = cv2.VideoCapture(0)
    last_label = ""
    label = ""
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        frame, hand_bbox = detect_hand(frame)
        if hand_bbox:
            xmin, ymin, xmax, ymax = hand_bbox
            hand_crop = frame[ymin:ymax, xmin:xmax]
            keypoints = extract_skeleton(hand_crop)
            if keypoints is not None:
                try:
                    label = predict_sign(keypoints)
                except:
                    label = "unknown"
                if label != last_label and label != "unknown":
                    last_label = label
                    texttospeech(label)
            cv2.putText(frame, f"Predicted: {label}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imshow("Sign Language Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

main = Tk()
main.title("Sign Language Detection")
main.geometry("600x300")

Button(main, text="Start Webcam", command=signfromWebcam).pack(pady=50)

main.mainloop()
