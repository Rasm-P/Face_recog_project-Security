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


print("[INFO] loading face detector...")
protoPath = "C:/Users/rasmu/Desktop/Face_recog_project-Security/spoofing_countermeasures/face_detector/deploy.prototxt"
modelPath = "C:/Users/rasmu/Desktop/Face_recog_project-Security/spoofing_countermeasures/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


def liveliness_detector():

    print("[INFO] loading liveness detector...")
    model = load_model("C:/Users/rasmu/Desktop/Face_recog_project-Security/spoofing_countermeasures/liveness/livenessRasmus2.model")
    le = pickle.loads(open("C:/Users/rasmu/Desktop/Face_recog_project-Security/spoofing_countermeasures/liveness/le.pickle", "rb").read())

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            # Face detection confidence
            if confidence > 0.8:

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

                    # Liveness predictions
                    preds = model.predict(face)[0]
                    print("Prediction: ", preds)
                    j = np.argmax(preds)
                    label = le.classes_[j]

                    label = "{}: {:.4f}".format(label, preds[j])
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                except:
                    continue

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    liveliness_detector()
