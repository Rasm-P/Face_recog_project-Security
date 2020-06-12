import unittest
from pathlib import Path
import os
import cv2

class Tests(unittest.TestCase):
    def test_imports(self):
        import cv2
        import keras
        import click
        import configparser
        import matplotlib
        import sklearn
        import PIL
        import scipy
        import tensorflow
        import dlib
        import imutils
        import keyboard
        import numpy
        import pandas

    def test_cv2_cascades(self):
        cv2dir = os.path.dirname(cv2.__file__)

        face_cascade_path = Path(f"{cv2dir}/data/haarcascade_frontalface_default.xml")
        eye_cascade_path = Path(f"{cv2dir}/data/haarcascade_eye.xml")
        smile_cascade_path = Path(f"{cv2dir}/data/haarcascade_smile.xml")

        self.assertTrue(face_cascade_path.exists())
        self.assertTrue(eye_cascade_path.exists())
        self.assertTrue(smile_cascade_path.exists())

    def test_cv2_cam_connection(self):
        cam = cv2.VideoCapture(0)
        self.assertTrue(cam.isOpened())
        cam.release()

