import os,sys
import numpy as np 
import cv2
import scipy
import pickle as pkl 
import random
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn.neural_network as nn 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def train_coarse_model(X,y):
	X_train, X_test, y_train, y_test = train_test_split( X, y, 
		test_size=0.2, random_state=42, shuffle=False)


	classifier =  MLPClassifier(hidden_layer_sizes = (256,), verbose=True, 
		tol=0.0001, max_iter=200)
	classifier.fit(X_train, y_train)
	acc = classifier.score(X_test,y_test)
	print "acc = %f" % acc

	with open("squeezenet_coarse_mlp_classifier_256.pkl", "w") as f:
		pkl.dump(classifier, f)


def save_encoders(lec, lef, ohec, ohef):
	pkl.dump(lec, open('lec.pkl', 'w'))
	pkl.dump(lef, open('lef.pkl', 'w'))
	pkl.dump(ohec, open('ohec.pkl', 'w'))
	pkl.dump(ohef, open('ohef.pkl', 'w'))


def read_encoders():
	lec = pkl.load(open('coarse_encodings/lec.pkl', 'r'))
	lef = pkl.load(open('coarse_encodings/lef.pkl', 'r'))
	ohec = pkl.load(open('coarse_encodings/ohec.pkl', 'r'))
	ohef = pkl.load(open('coarse_encodings/ohef.pkl', 'r'))
	return lec,lef,ohec,ohef


def one_hot_labels(image_labels, train_encoding=True):
	N = len(image_labels)
	coarse_labels = []
	fine_labels = []
	image_name = []
	for image_label in image_labels:
		image_name.append(image_label.split("/")[2])
		fine_labels.append(image_label.split("/")[0]+'@'+image_label.split("/")[1])
		coarse_labels.append(image_label.split("/")[0])

	if train_encoding == True:
		lec = LabelEncoder()
		lef = LabelEncoder()
		ohec = OneHotEncoder(categorical_features = [0])
		ohef = OneHotEncoder(categorical_features = [0])
		labels_coarse = lec.fit_transform(coarse_labels).reshape((N,1))
		labels_fine = lef.fit_transform(fine_labels).reshape((N,1))
		ohe_coarse_labels = ohec.fit_transform(labels_coarse).toarray()
		ohe_fine_labels = ohef.fit_transform(labels_fine).toarray()
		save_encoders(lec,lef,ohec,ohef)
	else:
		lec,lef,ohec,ohef = read_encoders()
		labels_coarse = lec.transform(coarse_labels).reshape((N,1))
		labels_fine = lef.transform(fine_labels).reshape((N,1))
		ohe_coarse_labels = ohec.transform(labels_coarse).toarray()
		ohe_fine_labels = ohef.transform(labels_fine).toarray()
	
	return ohe_coarse_labels, ohe_fine_labels


def main():

	feature_dictionary = pkl.load(open("features/squeezenet_features.pkl", "r"))
	image_labels = list(feature_dictionary.keys())
	feature_vector = np.array(feature_dictionary.values())
	print feature_vector.shape
	zipped = list(zip(image_labels, feature_vector))
	random.shuffle(zipped)
	image_labels, feature_vector = zip(*zipped)

	ohe_coarse_labels, ohe_fine_labels =  one_hot_labels(image_labels, False)
	print "Training...."
	train_coarse_model(feature_vector, ohe_coarse_labels)
	


if __name__ == '__main__':
	main()
	
