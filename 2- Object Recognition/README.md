# Description of the files for object categorizzation

## bilinear_cnn.py

Command to run : `python bilinear_cnn.py`
This files combines the two cnn outputs as proposed in bilinear cnn model and trains them for classification. The feature vector that we have used are those of squeezenet and resnet-18 CNN models.

##  coarse_grained.py

Command to run: `python coarse_grained.py`
This file is used to train a MLP classifier given the feature vector, feature vector can be sift, orb , resnet etc.

## fine_grained.py

Command to run: `python fine_grained.py`
The file is used to train either a single MLP classifier or multiple MLP Classifiers each for a class given a feature vector set. The training can also be done by concatenating the one hot coarse class vector along with the given feature vector.

## hasher.py

Command to run: `python hasher.py`
Code written to hash the content of file.

## prediction.py

Command to run: `python prediction.py`
Predicts the output given a set of feature vectors for the test images and stores the output to a file `output.txt`

## read.py

Command to run: `python read.py`
Code to read the training images given a folder which contains various coaarse classes, and each coarse class containing multiple fine classes. Returns the imagename, its coarse class and fine class along with their openCV arrays.