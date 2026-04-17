import cv2 
import numpy as np 
import mediapipe as mp 
from tensorflow.keras.models import load_model 
from ultralytics import YOLO 
from tkinter import * 
from gtts import gTTS 
from playsound import playsound 
import os 
import time 

# ------------------ Labels ------------------ 
sign_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Apple', 'Can', 'Get', 'Good', 'Give me a call', 'I love you', 'I want money', 'I want to go to the washroom', 'Please stop', 'Thank you very much'] 

# ------------------ Load Models ------------------ 
STGCN_MODEL_PATH = "SignLanguage\model\stgcn_model.h5" 
stgcn_model = load_model(STGCN_MODEL_PATH) 
print("ST-GCN Model Loaded") 

YOLO_MODEL_PATH = "SignLanguage\model\yolo11_best.pt" 
yolo_model = YOLO(YOLO_MODEL_PATH) 
print("YOLO11 Model Loaded") 

# ------------------ Mediapipe Pose ------------------ 
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose() 

# ------------------ Text-to-Speech ------------------ 
def texttospeech(text, filename="voice"): 
    filename = filename + '.mp3' 
    if os.path.exists(filename): 
        os.remove(filename) 
    tts = gTTS(text=text, lang="en", slow=False) 
    tts.save(filename) 
    playsound(filename) 
    os.remove(filename) 

# ------------------ Extract Skeleton ------------------ 
def extract_skeleton(frame): 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = pose.process(rgb) 
    if results.pose_landmarks: 
        keypoints = [] 
        for lm in results.pose_landmarks.landmark: 
            keypoints.append([lm.x, lm.y, lm.z]) 
        return np.array(keypoints) 
    return None 

# ------------------ Predict Sign ------------------ 
def predict_sign(keypoints): 
    x = np.expand_dims(keypoints, axis=0) # batch 
    x = np.expand_dims(x, axis=0) # time dimension = 1 
    x = np.expand_dims(x, axis=-1) # channel 
    prediction = stgcn_model.predict(x, verbose=0) 
    label_id = np.argmax(prediction) 
    return sign_labels[label_id] 

# ------------------ YOLO Detection ------------------ 
CONFIDENCE_THRESHOLD = 0.50 
GREEN = (0, 255, 0) 
def detect_hand(frame): 
    detections = yolo_model(frame)[0] 
    hand_bbox = None 
    for data in detections.boxes.data.tolist(): 
        confidence = data[4] 
        cls_id = data[5] 
        if float(confidence) >= CONFIDENCE_THRESHOLD: 
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3]) 
            hand_bbox = (xmin, ymin, xmax, ymax) 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2) 
            cv2.putText(frame, f"{sign_labels[int(cls_id)]}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2) 
            break 
    return frame, hand_bbox 

# ------------------ Webcam Function ------------------ 
def signfromWebcam(): 
    camera = cv2.VideoCapture(0) 
    last_label = "" 
    while True: 
        ret, frame = camera.read() 
        if not ret: 
            break 
        # Detect hand first 
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
                    print("Detected Sign:", label) 
                    texttospeech(label) 
            cv2.putText(frame, f"Predicted: {label}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2) 
        cv2.imshow("Sign Language Detection", frame) 
        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break 
    camera.release() 
    cv2.destroyAllWindows() 

# ------------------ GUI ------------------ 
main = Tk() 
main.title("Sign Language Detection") 
main.geometry("600x300") 
font = ('times', 16, 'bold') 
title = Label(main, text='Sign Language Detection (YOLO + ST-GCN)') 
title.config(bg='chocolate', fg='black') 
title.config(font=font) 
title.config(height=3, width=120) 
title.place(x=0, y=5) 

font1 = ('times', 13, 'bold') 
camButton = Button(main, text="Hand Sign from Webcam", command=signfromWebcam) 
camButton.place(x=200, y=150) 
camButton.config(font=font1, fg='black') 

main.config(bg='light salmon') 
main.mainloop()