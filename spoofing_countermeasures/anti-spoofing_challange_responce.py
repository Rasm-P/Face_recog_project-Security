import cv2
import os, os.path
import tkinter as tk
from tkinter import simpledialog
import keyboard


cv2dir = str(os.path.dirname(cv2.__file__))

face_cascade_path = f"{cv2dir}/data/haarcascade_frontalface_default.xml"
left_eye_cascade_path = f"{cv2dir}/data/haarcascade_lefteye_2splits.xml"
right_eye_cascade_path = f"{cv2dir}/data/haarcascade_righteye_2splits.xml"
smile_cascade_path = f"{cv2dir}/data/haarcascade_smile.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_path)
right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)


def execute_tracking():   
    ROOT = tk.Tk()
    ROOT.withdraw()

    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        ret, frame=cam.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canvas = detect(gray,frame)
            cv2.imshow('Face recognition',canvas)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        else:
            break
                
    cam.release()
    cv2.destroyAllWindows()


def detect(gray, frame):
    label = False

    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]  
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

        left_eye = left_eye_cascade.detectMultiScale(roi_gray, 1.07, 22)
        print("left_eye: ", left_eye)
        for (ex,ey,ew,eh) in left_eye:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0, 0, 255), 1)

        right_eye = right_eye_cascade.detectMultiScale(roi_gray, 1.07, 22)
        print("right_eye: ", right_eye)
        for (ex,ey,ew,eh) in left_eye:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0, 0, 255), 1)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.3, 30)
        print("smile: ", smile)
        for (xs,ys,ws,hs) in smile:
            cv2.rectangle(roi_color, (xs,ys), (xs+ws, ys+hs),(255, 0, 0), 1)

        if (len(left_eye) != 0) & (len(right_eye) != 0) & (len(smile) != 0):
            label = True
        else:
            label = False

        label = str(label)
        cv2.putText(frame, label, (x , y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


if __name__ == "__main__":
    execute_tracking()