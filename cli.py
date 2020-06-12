import click
from facial_tracking.facial_tracking import execute_tracking
from facial_tracking.recognition import execute_recognition
from logic.recognition_file import loadrecog
from logic.classify_known_faces import train_classifier, classify_people_from_path, classify_single_image
from logic.write_to_csv import plot_csv_data
from facial_tracking.videorecog import execute_videorecog


@click.group()
def frecog():
    pass


@frecog.command()
@click.option("--track", "-t", is_flag=True)
@click.option("--recognize", "-r", is_flag=True)
@click.argument('model', required=False)
def run(track, recognize, model):
    '''
    Primary executable functionalities
    E.g: frecog run -t
    E.g: frecog run -r
    '''
    if track:
        execute_tracking()
    if recognize:
        execute_recognition(model=model)


@frecog.command()
@click.option("--train", "-tr", type=click.Choice(['small', 'large']))
@click.argument('value', required=False)
def trainer(train, value):
    '''
    Train facial classifier
    E.g: frecog trainer -tr large 2
    '''
    if train:
        if value != None:
            train_classifier(number_neighbors=int(value), model=train)
        else:
            train_classifier(model=train)


@frecog.command()
@click.option('--path','-p')
@click.option("--single", "-s")
def classify(single, path):
    '''
    Classifies unknown pictures in a directory using the knn_model
    E.g. path: frecog classify -p facerec/unknown_faces
    E.g. single: frecog classify -s facerec/unknown_faces/eka01.jpg
    '''
    if path:
        classify_people_from_path(path)
    elif single:
        classify_single_image(single)


@frecog.command()
@click.option("--csv", "-c", nargs=2, required=True)
@click.option("--benchmark", '-b')
def graph(csv, benchmark):
    '''
    Plots a graph of the linalg norm distance where csv[1] is name and csv[0] is file name
    E.g. csv: frecog graph -c rasmusb1.csv Rasmus
    E.g. benchmark: frecog graph -c rasmusb1.csv Rasmus -b large
    '''
    if csv:
        if benchmark:
            execute_recognition(model=benchmark, benchmark=csv[0])
        plot_csv_data(csv[1], csv[0])


@frecog.command()   
@click.option('--movie' , '-m', type=click.Choice(['small', 'large']))
@click.argument('path', required=False)
def play(movie, path):
    '''
    Face recognises on a video
    E.g: frecog play -m small
    E.g: frecog play -m small ./vids/pathTest.mp4
    '''
    if movie:
        execute_videorecog(movie, path)


@frecog.command()
@click.option("--folder", "-f", type=click.Choice(['small', 'large']))
@click.argument('path', required=False)
def fold(folder, path):
    '''
    Face recognises on a folder of pictures
    E.g: frecog fold -f large
    E.g: frecog fold -f large ./facerec/unknown_faces
    '''
    if folder:
        loadrecog(folder, path)