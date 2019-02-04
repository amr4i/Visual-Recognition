import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2
import os
from read import load_images
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


def categorize():
	NUM_CLASSES = 16
	IMAGE_RESIZE = 224
	BATCH_SIZE_TESTING = 1
	LOSS_METRICS = ['accuracy']


	RESNET50_POOLING_AVERAGE = 'avg'
	DENSE_LAYER_ACTIVATION = 'softmax'
	OBJECTIVE_FUNCTION = 'categorical_crossentropy'
	image_size = IMAGE_RESIZE
	data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
	resnet_weights_path = '../resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



	model = Sequential()
	model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = resnet_weights_path))
	model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False

	# model.summary()
	sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
	model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

	model.load_weights("../resnet_model_1.hdf5")


	test_generator = data_generator.flow_from_directory(
    directory = '../sample_test',
    target_size = (image_size, image_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
	)
	print test_generator

	pred = model.predict(test_generator, steps = len(test_generator), verbose =  1)

	print pred 

if __name__ == '__main__':

	categorize( )