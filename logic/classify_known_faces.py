import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image
import dlib
import numpy as np
from pkg_resources import resource_filename
import click
import logic.logconfig as log
import configparser
import cv2


model_save_path = os.path.join("face_learning_model/models/knn_model.clf")
train_dir = "facerec/known_faces"
model_path = os.path.join("face_learning_model/models/knn_model.clf")


# Trained Model made to find 5 landmark points in a picture. These points can be used for linear classification.
def five_point_model_location():
    return os.path.join("face_learning_model/models/shape_predictor_5_face_landmarks.dat")


# Trained Model made to find 68 landmark points in a picture. These points can be used for linear classification.
def sixty_eight_point_model_location():
    return os.path.join("face_learning_model/models/shape_predictor_68_face_landmarks.dat")


 # dlib's face recognition tool. This tool maps an image of a human face to vector space checking if their distance is small enough.
def recognition_model_location():
    return os.path.join("face_learning_model/models/dlib_face_recognition_resnet_model_v1.dat")


# CNN Model trained for face detection.
def cnn_model_location():
    return os.path.join("face_learning_model/models/mmod_human_face_detector.dat")


# Returns dlip's default face detector.
face_detector = dlib.get_frontal_face_detector()

five_point_model = five_point_model_location()
five_point_predictor = dlib.shape_predictor(five_point_model)

sixty_eight_point_model = sixty_eight_point_model_location()
sixty_eight_point_predictor = dlib.shape_predictor(sixty_eight_point_model)

face_recognition_model = recognition_model_location()
facial_encoder = dlib.face_recognition_model_v1(face_recognition_model)

cnn_face_detection_model = cnn_model_location()
cnn_facial_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)


# Train Knn nearest neighbor algorithm on top of dlib landmarks and folder classification.
# Sklearn ball_tree neighbors search algorithm for fast indexing.
def train_classifier(number_neighbors=None, knn_algorithm='ball_tree',model='small'):
    
    X_train = []
    y_train = []

    with click.progressbar(os.listdir(train_dir)) as dir_:
        for class_dir in dir_:
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            for img_path in find_images_in_folder(os.path.join(train_dir, class_dir)):
                image = load_image_from_path(img_path)
                face_bounding_boxes = find_face_locations(image)

                if len(face_bounding_boxes) != 1:
                    print("No people in the picture: ", img_path)
                else:
                    X_train.append(find_facial_encodings(image, face_bounding_boxes, 1, model)[0])
                    y_train.append(class_dir)

    if number_neighbors is None:
        number_neighbors = int(round(math.sqrt(len(X_train))))
        print("Number of neighbors: ", number_neighbors)

    knn_clf_model = neighbors.KNeighborsClassifier(n_neighbors=number_neighbors, algorithm=knn_algorithm, weights='distance')
    knn_clf_model.fit(X_train, y_train)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf_model, f)

    return knn_clf_model


# Finds all image paths in a folder and returns them as a list.
def find_images_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]


# Perdicts the name of each face in a picture using our trained k-nearest-neighbor classification model from above.
def predict(image, distance_threshold=0.6, model='small', n_neighbors=1):

    with open(model_path, 'rb') as f:
        knn_clf_model = pickle.load(f)

    face_locations = find_face_locations(image)

    if len(face_locations) == 0:
        return []

    facial_encodings = find_facial_encodings(image, face_locations, 1, model)

    closest_distances = knn_clf_model.kneighbors(facial_encodings, n_neighbors=n_neighbors)
    matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    prediction = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf_model.predict(facial_encodings), face_locations, matches)]
    
    return prediction


# Returns a list of faical encodings as numpy arrays.
# Uses dlib recognition tool to produce facial encodings for each landmark set.
def find_facial_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    raw_facial_landmarks = find_raw_facial_landmarks(face_image, known_face_locations, model)
    return [np.array(facial_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_facial_landmarks]


# Returns raw facial landmarks based on a facial image.
def find_raw_facial_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = find_raw_face_locations(face_image)
    else:
        face_locations = [location_rectangle(face_location) for face_location in face_locations]

    pose_predictor = sixty_eight_point_predictor

    if model == "small":
        pose_predictor = five_point_predictor

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


# Returns a dlib rectangle object representing the rectangular area of an image.
def location_rectangle(dimensions):
    return dlib.rectangle(dimensions[3], dimensions[0], dimensions[1], dimensions[2])


# Returns a list of trimmed face location dimentions.
# hog uses the default dlib recognition model
def find_face_locations(image, number_upsample=1, model="hog"):
    if model == "cnn":
        return [trim_rectangle_dimensions(rectangle_dimensions(face.rect), image.shape) for face in find_raw_face_locations(image, number_upsample, "cnn")]
    else:
        return [trim_rectangle_dimensions(rectangle_dimensions(face), image.shape) for face in find_raw_face_locations(image, number_upsample, model)]


# Returns raw face locations for an image.
# hog uses the default dlib recognition model.
def find_raw_face_locations(image, number_upsample=1, model="hog"):
    if model == "cnn":
        return cnn_facial_detector(image, number_upsample)
    else:
        return face_detector(image, number_upsample)


# Returns face dimensions of a raw face location from find_raw_face_locations().
def rectangle_dimensions(rectangle):
    return rectangle.top(), rectangle.right(), rectangle.bottom(), rectangle.left()


# Trims the facial dimensions from rectangle_dimensions() to their max and min values for each dimension.
def trim_rectangle_dimensions(rectangle_dimension, image_shape):
    return max(rectangle_dimension[0], 0), min(rectangle_dimension[1], image_shape[1]), min(rectangle_dimension[2], image_shape[0]), max(rectangle_dimension[3], 0)


# Returns an image from a file path as RGB.
def load_image_from_path(file, mode='RGB'):
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


# Classifies people in images from a given path.
# Uses the predictor based on our knn model, that returns a list of matches in the picture.
# Prints out each prediction as well as it's dimensions.
def classify_people_from_path(picture_path):
    for image_file in os.listdir(picture_path):
        full_file_path = os.path.join(picture_path, image_file)

        print("Looking for faces in {}".format(full_file_path))
        image = load_image_from_path(full_file_path)
        predictions = predict(image)

        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {},{}, {})".format(name, top, right, bottom, left))


# Classifies people in a single image from a given path.
# Uses the predictor based on our knn model, that returns a list of matches in the picture.
# Prints out each prediction in the image as well as it's dimensions.
def classify_single_image(full_picture_path):
    image = load_image_from_path(full_picture_path)
    predictions = predict(image)

    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {},{}, {})".format(name, top, right, bottom, left))


# Loads known faces from the known faces directory, and returns a list of facial encodings as well as a list of their names.
def loading_known_faces(model):
    logger = log.logger
    conf = configparser.ConfigParser()
    conf.read("./settings/configuration.ini")
    frecog_conf = conf["FACE_RECOGNITION"]

    logger.info("loading known faces and names...\n")
    print('Loading known faces and names...')

    known_faces_list = []
    known_names_list = []

    with click.progressbar(os.listdir(frecog_conf["KnownFacesDir"])) as faces:
        for name in faces:

            logger.info(len(os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}")))
            print(len(os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}")))

            for filename in os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}"):
                image = load_image_from_path(f"{frecog_conf['KnownFacesDir']}/{name}/{filename}")
                encoding = find_facial_encodings(image,None,1,model)
                if len(encoding) > 0:
                    encoding = encoding[0]
                else:
                    logger.info("No faces found in the image!")
                    print("No faces found in the image!")
                    pass
                known_faces_list.append(encoding)
                known_names_list.append(name)

    return known_faces_list, known_names_list


# Returns a list of comparison data, as well as the raw comparison values.
# Everything above the distance tolerance is not add to the list.
def face_comparison_list(known_face_encodings, face_encoding_to_check, recognition_tolerance):
    values = linear_face_distance(known_face_encodings, face_encoding_to_check)
    return list(values <= recognition_tolerance), values


# Function is used to calculate vector norms. Norms in mathematics are used to characterize positive scalar values that can be used for comparison with other norms.
# linear algebra. Norm of (a-b), this is to calculate the distance between a and b.
def linear_face_distance(face_encodings_list, face_to_compare):
    if len(face_encodings_list) == 0:
        return np.empty((0))
    linarg = np.linalg.norm(face_encodings_list - face_to_compare, axis=1)
    return linarg


# Returns a color depending on the given name
# For every letter in the first 3 characters, ord will return a Unicode point, that we then -97 and multiply by 8, to create an aray of 3 int values, rgb.
def convert_name_to_color(name):
    name_color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return name_color


# Rescales a frame by a certain percentage
# Interpolation is a method of constructing new data points within the range of a set of known data points. A way to scale it down.
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)