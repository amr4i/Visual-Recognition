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

def modify_feature_vector(feature_vector, ohe_coarse_labels):
	ohe_coarse_labels = np.array(ohe_coarse_labels)
	updated = []
	for i, vec in enumerate(feature_vector):
		updated.append(np.concatenate((vec,ohe_coarse_labels[i])))
	return np.array(updated)	

def train_fine_model(X,y,i = 5):
	coarse_classes = ['aircrafts_', 'birds_', 'cars_', 'dogs_', 'flowers_', '']
	print y.shape

	X_train, X_test, y_train, y_test = train_test_split( X, y, 
		test_size=0.2, random_state=42, shuffle=False)


	classifier =  MLPClassifier(hidden_layer_sizes = (256,), verbose=True, 
		tol=0.0001, max_iter=200)
	classifier.fit(X_train, y_train)
	acc = classifier.score(X_test,y_test)
	print "acc = %f" % acc

	with open(coarse_classes[i] + "resnet_fine_distinct_mlp_classifier_256.pkl", "w") as f:
		pkl.dump(classifier, f)


def segregate_coarse_classes(image_labels, feature_vector):

	feature_aircrafts = []
	labels_aircrafts = []
	feature_birds = []
	labels_birds = []
	feature_cars = []
	labels_cars = []
	feature_dogs = []
	labels_dogs = []
	feature_flowers = []
	labels_flowers = []

	for i, image_label in enumerate(image_labels):
		coarse_class = image_label.split("/")[0]
		if coarse_class == 'aircrafts':
			feature_aircrafts.append(feature_vector[i])
			labels_aircrafts.append(image_label)
		elif coarse_class == 'birds':
			feature_birds.append(feature_vector[i])
			labels_birds.append(image_label)
		elif coarse_class == 'cars':
			feature_cars.append(feature_vector[i])
			labels_cars.append(image_label)
		elif coarse_class == 'dogs':
			feature_dogs.append(feature_vector[i])
			labels_dogs.append(image_label)
		elif coarse_class == 'flowers':
			feature_flowers.append(feature_vector[i])
			labels_flowers.append(image_label)

	feature_aircrafts = np.array(feature_aircrafts)
	feature_birds = np.array(feature_birds)
	feature_cars = np.array(feature_cars)
	feature_dogs = np.array(feature_dogs)
	feature_flowers = np.array(feature_flowers)
	data = [list(zip(labels_aircrafts, feature_aircrafts)),
		list(zip(labels_birds, feature_birds)),
		list(zip(labels_cars, feature_cars)),
		list(zip(labels_dogs, feature_dogs)),
		list(zip(labels_flowers, feature_flowers))]
	return data



def save_encoders(lec, lef, ohec, ohef, i=5):
	coarse_classes = ['aircrafts_', 'birds_', 'cars_', 'dogs_', 'flowers_', '']
	pkl.dump(lec, open(coarse_classes[i] + 'lec.pkl', 'w'))
	pkl.dump(lef, open(coarse_classes[i] + 'lef.pkl', 'w'))
	pkl.dump(ohec, open(coarse_classes[i] + 'ohec.pkl', 'w'))
	pkl.dump(ohef, open(coarse_classes[i] + 'ohef.pkl', 'w'))


def read_encoders(i):
	coarse_classes = ['aircrafts_', 'birds_', 'cars_', 'dogs_', 'flowers_', '']
	lec = pkl.load(open('coarse_encodings/soft/'+coarse_classes[i] +'lec.pkl', 'r'))
	lef = pkl.load(open('coarse_encodings/soft/'+coarse_classes[i] +'lef.pkl', 'r'))
	ohec = pkl.load(open('coarse_encodings/soft/'+coarse_classes[i] +'ohec.pkl', 'r'))
	ohef = pkl.load(open('coarse_encodings/soft/'+coarse_classes[i] +'ohef.pkl', 'r'))
	return lec,lef,ohec,ohef


def one_hot_labels(image_labels, train_encoding=True, i = 5):
	coarse_classes = ['aircrafts_', 'birds_', 'cars_', 'dogs_', 'flowers_', '']
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
		save_encoders(lec,lef,ohec,ohef,i)
	else:
		lec,lef,ohec,ohef = read_encoders(i)
		labels_coarse = lec.transform(coarse_labels).reshape((N,1))
		labels_fine = lef.transform(fine_labels).reshape((N,1))
		ohe_coarse_labels = ohec.transform(labels_coarse).toarray()
		ohe_fine_labels = ohef.transform(labels_fine).toarray()
	
	return ohe_coarse_labels, ohe_fine_labels


def main():

	feature_dictionary = pkl.load(open("features/resnet18_features.pkl", "r"))
	image_labels = list(feature_dictionary.keys())
	feature_vector = np.array(feature_dictionary.values())

	# print feature_vector.shape

	# ohe_coarse_labels,ohe_fine_labels = one_hot_labels(image_labels, False)
	# feature_vector = modify_feature_vector(feature_vector, ohe_coarse_labels)

	# print feature_vector.shape

	# zipped = list(zip(image_labels, feature_vector))
	# random.shuffle(zipped)
	# image_labels, feature_vector = zip(*zipped)
	# ohe_coarse_labels, ohe_fine_labels =  one_hot_labels(image_labels, False)
	# print "Training...."
	# train_fine_model(feature_vector, ohe_fine_labels, 5)


	zipped_images = segregate_coarse_classes(image_labels, feature_vector)

	for i, zipped in enumerate(zipped_images):

		random.shuffle(zipped)
		image_labels, feature_vector = zip(*zipped)
		# print image_labels

		ohe_coarse_labels, ohe_fine_labels =  one_hot_labels(image_labels, False, i)
		print "Training...."
		train_fine_model(feature_vector, ohe_fine_labels, i)





if __name__ == '__main__':
	main()
	
