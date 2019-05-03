import os,sys
import numpy as np 
import cv2
import scipy
import pickle as pkl 
import random
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from fine_grained import modify_feature_vector

def invert_coarse(encoding):
	lec = pkl.load(open('coarse_encodings/lec.pkl', 'r'))
	ohec = pkl.load(open('coarse_encodings/ohec.pkl', 'r'))
	inv = ohec.inverse_transform(np.array(encoding))
	return lec.inverse_transform(inv.astype(int)).tolist()

def invert_fine(encoding):
	lef = pkl.load(open('coarse_encodings/lef.pkl', 'r'))
	ohef = pkl.load(open('coarse_encodings/ohef.pkl', 'r'))
	inv = ohef.inverse_transform(np.array(encoding))
	return lef.inverse_transform(inv.astype(int)).tolist()

def invert_distinct_fine(encoding, coarse_classes):

	fine_class_name = []

	for enc, cc in zip(encoding, coarse_classes):
		lef = pkl.load(open('coarse_encodings/soft/' + cc + '_lef.pkl', 'r'))
		ohef = pkl.load(open('coarse_encodings/soft/' + cc + '_ohef.pkl', 'r'))
		inv = ohef.inverse_transform(np.array([enc]))
		inv_name = lef.inverse_transform(inv.astype(int)).tolist()
		fine_class_name.append(inv_name[0])
	return fine_class_name

	

def predict_coarse(X):
	coarse_classes = []
	coarse_model = pkl.load(open("models/resnet_coarse_mlp_classifier_256.pkl", "r"))
	for feature in X:
		# print coarse_model.predict([feature])
		coarse_classes.append(coarse_model.predict([feature]).flatten())
	return coarse_classes

def predict_fine(X):
	fine_classes = []
	fine_model = pkl.load(open("models/resnet_fine_concatenated_mlp_classifier_256.pkl", "r"))
	for feature in X:
		fine_classes.append(fine_model.predict([feature]).flatten())
	return fine_classes

def predict_fine_classes(X, coarse_classes):
	fine_classes = []
	coarse_ref = ['aircrafts_', 'birds_', 'cars_', 'dogs_', 'flowers_', '']

	for feature, i in zip(X, coarse_classes):
		# if i == 'aircrafts':
		# 	fine_model = pkl.load(open("models/fine_resnet_concatenated/aircrafts_resnet_fine_", "r"))
		# elif i == 'birds':
		# 	fine_model = pkl.load(open("model/", "r"))
		# elif i == 'cars':
		# 	fine_model = pkl.load(open("model/", "r"))
		# elif i == 'dogs':
		# 	fine_model = pkl.load(open("model/", "r"))
		# elif i == 'flowers':
		# 	fine_model = pkl.load(open("model/", "r"))
		fine_model = pkl.load(open("models/fine_resnet_distinct/" + i + "_resnet_fine_distinct_mlp_classifier_256.pkl"))
		fine_classes.append(fine_model.predict([feature]).flatten())

	return fine_classes


def write_everything_to_file(images, coarse, fine):
	combined = []
	for i, c, f in zip(images, coarse, fine):
		combined.append(i+" "+ c+ " " + f)
	with open ('output.txt', 'w') as f:
		for item in combined:
			f.write("%s\n" % item)

def accuracy(pred_coarse, pred_fine, coarse, fine):
	count = 0;
	
	for pc, pf, c, f in zip(pred_coarse, pred_fine, coarse, fine):
		if pc == c and pf == f: 	
			count += 1
	return count/len(coarse)


def main():

	feature_dictionary = pkl.load(open("features/test_resnet18_features.pkl", "r"))
	image_names = list(feature_dictionary.keys())
	# print image_names
	feature_vector = np.array(feature_dictionary.values())
	coarse_classes = predict_coarse(feature_vector)
	# print coarse_classes
	coarse_class_name = invert_coarse(coarse_classes)

	# print coarse_class_name
	# feature_vector = modify_feature_vector(feature_vector, coarse_classes)

	# fine_class = predict_fine(feature_vector)
	# fine_class_name = invert_fine(fine_class)
	# print fine_class_name

	fine_class = predict_fine_classes(feature_vector, coarse_class_name)
	# fine_class_name = invert_fine(fine_class)
	fine_class_name = invert_distinct_fine(fine_class, coarse_class_name)

	# print fine_class_name

	write_everything_to_file(image_names, coarse_class_name, fine_class_name)

	# accuracy(coarse_class_name, fine_class_name, true_coarse, fine_coarse)
	


if __name__ == '__main__':
	main()
	
