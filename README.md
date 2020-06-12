# Facial recognition project
Agreeable Government, dat4sem2020spring-python

### Made by:
- cph-rp134 Rasmus Barfod Prætorius
- cph-hw98 Henning Wiberg
- cph-lb269 Lukas Bjørnvad

### Table of Contents:
- [Description](https://github.com/Mutestock/face_recog_project#description)
- [Technologies](https://github.com/Mutestock/face_recog_project#technologies)
- [Installation](https://github.com/Mutestock/face_recog_project#installation)
- [Disposition](https://github.com/Mutestock/face_recog_project#disposition)
- [Training our own neural network](https://github.com/Mutestock/face_recog_project#training-our-own-neural-network)
- [How to use the framework](https://github.com/Mutestock/face_recog_project#how-to-use-the-framework)
- [Technologies we would have added if we had more time](https://github.com/Mutestock/face_recog_project#technologies-we-would-have-added-if-we-had-more-time)

# Description
This framework provides cli commands for facial recognition and tracking functions to detect users based on webcam and other media footage such as people in photos and videos. The framework utilizes stored facial images for the recognition, and it is possible for the user to add more through the tracking feature. The project has a set of cli commands for each main function. The project requires a moderate amount of technical knowledge to operate.
 
Within the process of recognition, we have all so trained own Convolutional, Siamese, and using Convolutional, Siamese, and VGG-Face pre-calibrated neural network models as examples. This primarily involves examples on how to train our own neural networks with Keras and TenserFlow deep learning. For the recognition itself, we have made use of dlib’s pre-trained models for face detectors, facial landmark predictors and recognition models for higher recognition accuracy within the actual framework demo.

# Technologies
- Data processing
	- Neural networks, Deep learning
 		- Keras
	- Machine Learning
		- TensorFlow
- OpenCV
- Data wrangling
	- Image processing
	- One-hot labels
	- Classification
- Data collection
	- Working with video capture
	- Working with CSV and plotting
- CLI

For specific requirements see:  [requirements.txt](https://github.com/Mutestock/face_recog_project/blob/master/requirements.txt).

# Installation
Clone the project, cd into the project.
Run `pip install --editable .`

Project is compatible with pipenv and can be activated with 'pipenv shell', 'pipenv lock', 'pipenv sync'. Change the python version in the pipfile so that it matches yours.

Project cli commands can be run with an 'frecog' or a 'python main.py' prefix, depending on how the requirements were installed.
cli commands can be found in cli.py and contains examples.

Please be advised, that dlib can be rather sensitive. Especially so on Windows. If errors pop up please check the load order with cmake and imutils. If this doesn't work, you may have to go on a long journey through the Visual Studio IDE installer.


To use the the facial classification feature, please download and extract the vgg_face.mat file from [here](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/), and place it in the face_recog_project\face_learning_model\vgg_face_matconvnet folder. You will afterwards be able to train the classifier in the command line using ‘frecog trainer -tr large 2’. This classifier can be used to train a k-nearest neighbor model for specific faces to obtain more accurate classification results.

# Disposition
The project will be focusing on the development of a facial recognition framework that with a high certainty can detect users based on webcam and other media footage.

### Summary:

We will develop our own facial recognition framework that can receive a picture of a person, then through a neural network match the face with known faces to recognize and confirm the person in the picture.

### Goals for functionality:
- Detect faces in a picture/frame
- Make out own neural network examples that with somewhat high certainty can recognize faces by matching them with known faces. This part requires a lot of machine learning and training with a large dataset. Try Convolutional, Siamese, and VGG-Face pre-calibrated model types.
- Use pre-trained models by dlib to display a more accurate use of facial recognition within the framework itself. Here pre-trained models for face detectors, facial landmark predictors and recognition models will be used.
- Use facial recognition to recognize faces within webcam footage as well as videos and pictures from local directories.
- Use a benchmark to test the recognition values and probabilities.
	- If the face in question is recognized, save the match accuracy to a csv file, and return the correct name of the verified person's face.
	- If the face is not recognized, run the function again and save the failed match accuracy to a csv file.
	- Visualize/plot the verification data from the csv.
- Make a feature that can detect a face and then save samples if that face to known faces.
- Create a facial classifier, that can be trained on top of the dlib models and provide enhanced probability readings on specific classified faces.

### Concepts and Focus Areas:

The concepts involved in this project regarding the python course and related technologies include the following entries:
- OpenCV and image processing
- CLI
- Neural networks, Deep learning
- Machine Learning
- Data processing
- Data wrangling
- Data collection, working with video capture
- Working with CSV and plotting

# Training our own neural network
Besides using dlib for high precision recognition, we also made a few notebook examples on how neural networks for facial recognition could be trained using tensorflow and keras as well as large collections of facial image data. Among our own attempts to successfully train a neural network, we made several different demos using Convolutional, Siamese, and VGG-Face model pre-calibrated weights. From these demo networks we trained our own models and tested them to see their accuracy and performance.

[Convolutional network example:](https://github.com/Mutestock/face_recog_project/blob/master/face_learning_model/Convolutional%20network.ipynb)
![unnamed (7)](https://user-images.githubusercontent.com/44894132/82733864-39ef1580-9d17-11ea-9d99-13036f888499.png)

[Siamese network example:](https://github.com/Mutestock/face_recog_project/blob/master/face_learning_model/Siamese%20network.ipynb)

![unnamed (8)](https://user-images.githubusercontent.com/44894132/82733865-3c516f80-9d17-11ea-8c65-ff2e32c7e530.png)

[Neural network with VGG weights:](https://github.com/Mutestock/face_recog_project/blob/master/face_learning_model/Neural%20network%20with%20VGG%20weigths.ipynb)
![unnamed (9)](https://user-images.githubusercontent.com/44894132/82733866-3e1b3300-9d17-11ea-89f9-bf6158cdc1a1.png)

# How to use the framework
Install everything you need by following the installation guide.
Open up a cmd, bash or terminal based in the root of the face_recog_project  project.
All functions can be found through a `frecog --help` command.
![unnamed](https://user-images.githubusercontent.com/44894132/82733420-df07ef00-9d13-11ea-83d6-5fa1fbc91972.png)

Note that all windows that pop up can be closed by pressing the X at the top or pressing "q". To force a stop presse "ctrl C"

### Facial tracking and recognition
If you want the framework to work with your face through a webcam, run the tracking command by typing `frecog run -t` and then press "r". This will prompt a pop-up that you have to fill out with your name. The function takes 5 pictures from the webcam and saves them in the ‘face_recog_project\facerec\known_faces’ directory.
![s](https://user-images.githubusercontent.com/44894132/82733439-019a0800-9d14-11ea-9486-61ec330b69a4.PNG)

The recognition command is run by writing `frecog run -r` followed by either "small" or "large", this will determine the size of the model that will be used. If nothing is added at the end of the command, the default which is "large" will be run with, thus making the process slower.

![unnamed (6)](https://user-images.githubusercontent.com/44894132/82733682-c26cb680-9d15-11ea-9a0f-d196949c7ffa.png)

By pressing ‘f’ you can toggle on the facial landmarks as either a 5 point or a 68 point representation of the picture, depending on the model size. These landmarks are the ones used by the dlib algorithm to determine the linar distance between two faces and ultimately recognize them.

![sdfsd](https://user-images.githubusercontent.com/44894132/82733440-02cb3500-9d14-11ea-95d8-3f09ae7da1f8.PNG)

### Training a facial knn classification model
The trainer command is used to train the models that does the recognition, it’s run by `frecog trainer -tr` followed by either "small" or "large" and then an integer that represents the number of neighbors in the classified regression.
![unnamed (5)](https://user-images.githubusercontent.com/44894132/82733413-dd3e2b80-9d13-11ea-8e96-b90f7b4ef3fe.png)

### Facial recognition through knn classification
The classify by path command is run by `frecog classify -p` and then a path to a folder containing pictures. It then prints out its results of the facial recognition on the pictures contained in the folder.
![unnamed (4)](https://user-images.githubusercontent.com/44894132/82733415-ddd6c200-9d13-11ea-8c44-cdf0c4de0062.png)

The classify single command is run by `frecog classify -s` and then a path to a picture. It then prints out the results of the facial recognition on the picture.
![unnamed (3)](https://user-images.githubusercontent.com/44894132/82733416-de6f5880-9d13-11ea-9ba5-cd73751d2c85.png)

### Recognition graph for benchmarking of values and false positives.
The graph command shows a graph over how sure the program is that the person that it’s looking for, also included in the graph, is the  false positives. The command is run by `frecog graph -c` then a csv file name and then the name of the person you are trying to recognize.

![unnamed (2)](https://user-images.githubusercontent.com/44894132/82733418-de6f5880-9d13-11ea-9e15-5c2b2e6cc196.png)

The graph benchmark command is given a name, then utilizes the web-cam to gather information with which it creates and shows a graph over how sure the program is that the person that it’s looking for. The command is run by `frecog graph -c` then a pre-existing csv file name or a new csv file name, then the name of the person you are trying to recognize, then “-b” and lastly a model(“large” or “small”).
![dsdfs](https://user-images.githubusercontent.com/44894132/82733444-052d8f00-9d14-11ea-968c-3aba45e197f7.PNG)

### Facial recognition of video footage
The play command runs facial recognition on a video file. The command is run by `frecog play -m`, then a model(“large” or “small”), then it can be given a path to a file(eks: `frecog play -m ./vids/pathTest.mp4`), if it isn’t given a path it will utilize a default mp4.

![dfd](https://user-images.githubusercontent.com/44894132/82733442-0494f880-9d14-11ea-83f2-e5f8dcf80861.PNG)

### Facial recognition of single images and from directories
The fold command takes a folder of images, then it uses facial recognition on them then it opens them in a window where the result and information is displayed. The command is run by `frecog fold -f`, then a model(“large” or “small”), then it can be given a path to a folder. If it doesn’t receive a path it runs on a default folder path.
![unnamed (1)](https://user-images.githubusercontent.com/44894132/82733419-df07ef00-9d13-11ea-8f75-f48e2323980e.png)

# Technologies we would have added if we had more time
- Python web services with flask
- Requests, Headers and Authentication
- Host the neural network training on an external server
- Deployment of framework to a droplet
