import numpy as np
import cv2
import joblib
import argparse
import os, os.path


cv2dir = str(os.path.dirname(cv2.__file__))
face_cascade_path = f"{cv2dir}/data/haarcascade_frontalface_default.xml"
model_path = "C:/Users/rasmu/Desktop/face_recog_project/spoofing_countermeasures/trained_models/replay_attack_trained_models/replay-attack_ycrcb_luv_extraTreesClassifier.pkl"


def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110)
    )
    return faces


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


def run_detection():
    # Load model
    clf = None
    try:
        clf = joblib.load(model_path)
    except IOError as e:
        print ("Error loading model")
        exit(0)

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print ("Error opening camera")
        exit(0)

    width = 320
    height = 240
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    # Initialize face detector
    faceCascade = cv2.CascadeClassifier(face_cascade_path)

    sample_number = 1
    count = 0
    measures = np.zeros(sample_number, dtype=np.float)

    while True:
        ret, img_bgr = cap.read()
        if ret is False:
            print ("Error grabbing frame from camera")
            break

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = detect_face(img_gray, faceCascade)

        measures[count%sample_number]=0

        point = (0,0)
        for i, (x, y, w, h) in enumerate(faces):

            roi = img_bgr[y:y+h, x:x+w]

            img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
            img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)

            feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
            feature_vector = feature_vector.reshape(1, len(feature_vector))

            # Spoofing prediction
            prediction = clf.predict_proba(feature_vector)
            prob = prediction[0][1]

            measures[count % sample_number] = prob

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

            point = (x, y-5)

            print (measures, np.mean(measures))
            if 0 not in measures:
                text = "True"

                # Prediction tolerance
                if np.mean(measures) > 0.7:
                    text = "False"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                                thickness=2, lineType=cv2.LINE_AA)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9,
                                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        count+=1
        cv2.imshow('img_rgb', img_bgr)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
