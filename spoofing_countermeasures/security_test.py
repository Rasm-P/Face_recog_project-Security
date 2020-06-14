from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import copyreg

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import os.path
import threading

import cv2
import os, os.path
import tkinter as tk
from tkinter import simpledialog
import keyboard


anti_spoofing_cnn = False
anti_spoofing_eye_blink = False
anti_spoofing_challenge = False  

label = False

count = 0
stop_thread = False

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0


print("[INFO] loading face detector...")
protoPath = "C:/Users/rasmu/Desktop/Face_recog_project-Security/spoofing_countermeasures/face_detector/deploy.prototxt"
modelPath = "C:/Users/rasmu/Desktop/Face_recog_project-Security/spoofing_countermeasures/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

cv2dir = str(os.path.dirname(cv2.__file__))

face_cascade_path = f"{cv2dir}/data/haarcascade_frontalface_default.xml"
left_eye_cascade_path = f"{cv2dir}/data/haarcascade_lefteye_2splits.xml"
right_eye_cascade_path = f"{cv2dir}/data/haarcascade_righteye_2splits.xml"
smile_cascade_path = f"{cv2dir}/data/haarcascade_smile.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_path)
right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)


def liveliness_detector():
    global label
    global stop_thread

    print("[INFO] loading liveness detector...")
    model = load_model("C:/Users/rasmu/Desktop/Face_recog_project-Security/spoofing_countermeasures/liveness/livenessRasmus2.model")
    le = pickle.loads(open("C:/Users/rasmu/Desktop/Face_recog_project-Security/spoofing_countermeasures/liveness/le.pickle", "rb").read())


    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(sixty_eight_point_model_location())


    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    printit()


    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        anti_spoofing_liveness(frame,model,le)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        anti_spoofing_eye(frame, rects, gray, predictor)

        anti_spoofing_challenge_def(gray, frame)

        cv2.putText(frame, "CNN: {}".format(str(anti_spoofing_cnn)), (430, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Eye: {}".format(str(anti_spoofing_eye_blink)), (430, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Cha: {}".format(str(anti_spoofing_challenge)), (430, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if (anti_spoofing_cnn) & (anti_spoofing_eye_blink) & (anti_spoofing_challenge):
            label = True
        else:
            label = False

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            stop_thread = True
            break

    cv2.destroyAllWindows()
    vs.stop()


def anti_spoofing_liveness(frame,model,le):
    global anti_spoofing_cnn

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.9:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]

            try:
                face = cv2.resize(face, (32, 32))
            
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face)[0]

                j = np.argmax(preds)
                label = le.classes_[j]
                
                if str(label) == "b'real'":
                    anti_spoofing_cnn = True
                else:
                    anti_spoofing_cnn = False
            except:
                continue
            

def printit():
    global count
    global anti_spoofing_eye_blink

    thread = threading.Timer(1.0, printit)
    thread.start()
    count += 1
    if TOTAL/(count/60) > 15:
        anti_spoofing_eye_blink = True
    else:
        anti_spoofing_eye_blink = False

    if stop_thread:
      thread.cancel()


def sixty_eight_point_model_location():
    return os.path.join("face_learning_model/models/shape_predictor_68_face_landmarks.dat")


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear


def anti_spoofing_eye(frame, rects, gray, predictor):
    global COUNTER
    global TOTAL

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    for rect in rects:

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

            else:

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                COUNTER = 0


def anti_spoofing_challenge_def(gray, frame):
    global anti_spoofing_challenge

    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]  
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

        left_eye = left_eye_cascade.detectMultiScale(roi_gray, 1.07, 22)
 

        right_eye = right_eye_cascade.detectMultiScale(roi_gray, 1.07, 22)


        smile = smile_cascade.detectMultiScale(roi_gray, 1.3, 30)


        if (len(left_eye) != 0) & (len(right_eye) != 0) & (len(smile) != 0):
            anti_spoofing_challenge = True
        else:
            anti_spoofing_challenge = False

        cv2.putText(frame, str(label), (x , y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == "__main__":
    liveliness_detector()
