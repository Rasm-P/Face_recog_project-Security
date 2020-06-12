import cv2
import os, os.path
from logic.classify_known_faces import loading_known_faces, find_facial_encodings, find_face_locations, rescale_frame, load_image_from_path, face_comparison_list, convert_name_to_color
import numpy as np


TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2


def loadrecog(model, path=' '):
    print('Loading known faces...')
    known_faces, known_names = loading_known_faces(model)
    file_path = path
    if path == None:
        file_path="./facerec/testfaces"

    count = 0
    print('Processing unknown faces...')
    while True:
        size = len(os.listdir(file_path))
        for name in os.listdir(file_path):
            image = load_image_from_path(f"{file_path}/{name}")
            locations = find_face_locations(image)
            encodings = find_facial_encodings(image, locations,1,model)
            print(f', found {len(encodings)} face(s)')
            for face_encoding, face_location in zip(encodings, locations):
                results,values = face_comparison_list(known_faces, face_encoding, TOLERANCE)
                match = None

                best_linarg_value = np.argmin(values)
                if results[best_linarg_value]:
                    match = known_names[best_linarg_value]

                print(f' - {match} from {results}')
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                color = convert_name_to_color(match)
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height = image.shape[0] 
            width = image.shape[1] 
            if height > 490 or width > 690:
                image = rescale_frame(image, 50)
            cv2.imshow(name, image)
            
            count+=1
            print(f'{count} out of: {size}')
            
            if cv2.waitKey(3000) & 0xFF == ord("q"):
                cv2.destroyWindow(name)
            else:
                cv2.destroyWindow(name)
                
            
        if count>=size:
            break
    cv2.destroyAllWindows()