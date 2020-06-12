import cv2
import numpy as np
import logic.logconfig as log
from logic.classify_known_faces import face_comparison_list, convert_name_to_color, loading_known_faces, find_facial_encodings, find_face_locations, find_raw_facial_landmarks, rescale_frame
from imutils import face_utils
import keyboard
from pathlib import Path


logger = log.logger

show_facial_landmarks = False


def execute_videorecog(model="large", path =' '):
    global show_facial_landmarks

    recognition_tolerance = 0.6
    frame = 3
    font = 2
    videoname = path 
    if(videoname == None):
        cam = cv2.VideoCapture('./vids/test.mp4')
    else:
        if(Path(videoname).exists()):
            cam = cv2.VideoCapture(videoname)
        else:
            raise FileNotFoundError
    known_faces_list, known_names_list = loading_known_faces(model)
    # Sets the resolution to 240p
    cam.set(3, 426)
    cam.set(4, 240)
    logger.info("Processing unknown faces...")
    print('Processing unknown faces...')

    while True:
        if(cam.isOpened()):
            ret, image = cam.read()
            if ret:
                image = rescale_frame(image, percent=40)
                locations = find_face_locations(image)
                encodings = find_facial_encodings(image, locations,1,model)

                logger.info(f", found {len(encodings)} faces")
                print(f', found {len(encodings)} faces')

                for facial_encoding, face_location in zip(encodings, locations):
                    recognition_results, values = face_comparison_list(known_faces_list, facial_encoding, recognition_tolerance)

                    facial_match = "Unknown"

                    best_linarg_value = np.argmin(values)
                    linarg_value = (1-min(values))
                    if recognition_results[best_linarg_value]:
                        facial_match = known_names_list[best_linarg_value]

                    logger.info(f"Match found: {facial_match}, Linarg norm value: {linarg_value}%")
                    print(f"Match found: {facial_match}, Linarg norm value: {linarg_value}%")

                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])

                    name_color = convert_name_to_color(facial_match)

                    cv2.rectangle(image, top_left, bottom_right, name_color, frame)

                    if show_facial_landmarks:
                        for face_land in find_raw_facial_landmarks(image,None, model):
                            for (x, y) in face_utils.shape_to_np(face_land):
                                cv2.circle(image,(x, y),3,name_color,-1)

                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2] + 22)

                    cv2.rectangle(image, top_left, bottom_right, name_color, cv2.FILLED)

                    percent = linarg_value * 100
                    cv2.putText(image, facial_match + f' {"%.2f" % percent}%', (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), font)

                cv2.imshow('Running facial recognition', image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if keyboard.is_pressed('f'):
                    if show_facial_landmarks == True:
                        show_facial_landmarks = False
                    else:
                        show_facial_landmarks = True
            else:
                break

    cam.release()
    cv2.destroyAllWindows()


