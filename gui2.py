import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
import cv2
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json
import speech_recognition as sr

# Load pre-trained emotion detection model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect emotions and eye state from the camera feed
def Detect(camera, model, face_cascade, eye_cascade):
    recognizer = sr.Recognizer()  # Initialize the speech recognizer
    _, frame = camera.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        # Calculate face ratio
        face_ratio = w / h
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Face detection
        face_roi = gray_frame[y:y + h, x:x + w]
        resized_roi = cv2.resize(face_roi, (48, 48))
        pred = EMOTIONS_LIST[np.argmax(model.predict(resized_roi[np.newaxis, :, :, np.newaxis]))]
        cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Eye detection
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_color, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey + eh, ex:ex + ew]
            eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

            # Apply Hough Circle Transform for eye detection
            circles = cv2.HoughCircles(eye_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30,
                                       minRadius=0, maxRadius=30)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]

                    # Draw the circle
                    cv2.circle(eye_roi, center, radius, (0, 255, 0), 2)

                    # Calculate eye aspect ratio
                    aspect_ratio = radius / (2.0 * np.sqrt(2))

                    # Define thresholds for closed and open eyes based on aspect ratio
                    closed_eye_aspect_ratio_thresh = 0.3
                    open_eye_aspect_ratio_thresh = 0.5

                    if aspect_ratio < closed_eye_aspect_ratio_thresh:
                        cv2.putText(roi_color, "Closed", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    elif aspect_ratio > open_eye_aspect_ratio_thresh:
                        cv2.putText(roi_color, "Open", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(roi_color, "Uncertain", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print("Eyes not detected")
                cv2.putText(roi_color, "Eyes Not Detected", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Add text indicating face ratio
        cv2.putText(frame, f"Face Ratio: {face_ratio:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Voice recognition
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            caption = recognizer.recognize_google(audio)
            print(f"Caption: {caption}")
            cv2.putText(frame, f"Caption: {caption}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results: {e}")

    return frame


def update_frame():
    frame = Detect(camera, model, face_cascade, eye_cascade)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    display_label.imgtk = imgtk
    display_label.configure(image=imgtk)
    display_label.after(10, update_frame)


# Initialize Tkinter window
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load eye cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load emotion detection model
model = FacialExpressionModel("model_a1.json", "model_weights1.h5")
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create a label to display the camera feed
display_label = tk.Label(top)
display_label.pack()

# Initialize the camera
camera = cv2.VideoCapture(0)

# Update the frame
update_frame()

# Run the Tkinter event loop
top.mainloop()
