from setuptools import setup

setup(
    name='frecog',
    version='0.0.2',
    py_modules=['main'],
    install_requires=[
        'pandas == 1.0.3',
        'Click == 7.1.2',
        'nose2 == 0.9.2',
        'configparser == 5.0.0',
        'opencv-python == 4.2.0.34',
        'matplotlib == 3.2.1',
        'sklearn',
        'cmake == 3.17.2',
        'pillow == 7.1.2',
        'scipy == 1.4.1',
        'keras == 2.3.1',
        'h5py == 2.10.0',
        'tensorflow == 2.3.1',
        'imutils == 0.5.3',
        'dlib == 19.19.0',
        'keyboard == 0.13.5'
    ],
    author='TeamOne',
    entry_points='''
        [console_scripts]
        frecog=cli:frecog
    '''
)