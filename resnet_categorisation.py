import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
import sys
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
from keras.models import model_from_json


def categorize():
	NUM_CLASSES = 16
	IMAGE_RESIZE = 224
	BATCH_SIZE_TESTING = 1
	LOSS_METRICS = ['accuracy']
	image_path = sys.argv[1]


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

	sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
	model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)



	# model.load_weights("model_resnet.h5")
	model.load_weights("../working/resnet_model_20.hdf5")


	test_generator = data_generator.flow_from_directory(
    directory = '../test',
    target_size = (image_size, image_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
	)

	# for i in test_generator:
	#     idx = (test_generator.batch_index - 1) * test_generator.batch_size
	#     print(test_generator.filenames[idx : idx + test_generator.batch_size])
	
	class_names = [
	'3m_high_tack_spray_adhesive',
	'aunt_jemima_original_syrup',
	'campbells_chicken_noodle_soup',
	'cheez_it_white_cheddar',
	'cholula_chipotle_hot_sauce',
	'clif_crunch_chocolate_chip',
	'coca_cola_glass_bottle',
	'detergent',
	'expo_marker_red',
	'listerine_green',
	'nice_honey_roasted_almonds',
	'nutrigrain_apple_cinnamon',
	'palmolive_green',
	'pringles_bbq',
	'vo5_extra_body_volumizing_shampoo',
	'vo5_split_ends_anti_breakage_shampoo'	]	

	image_names = [str(f) for f in os.listdir('../test/sample_test')]
	image_names.sort()
	print image_names
	image_names = [f[:-4] + ".txt" for f in image_names]
	result_folder = "../test_results/"

	test_generator.reset()
	prediction = model.predict(test_generator, steps = len(test_generator), verbose =  1)
	print prediction
	for counter, instance in enumerate(prediction):
		class_indexes = instance.argsort()[::-1]
		predicted_classes = [class_names[i] for i in class_indexes]

		"""Ranking of Various Classes"""
		# with open(result_folder + image_names[counter], 'w') as f:
		# 	for item in predicted_classes:
		# 		f.write("%s\n" % item)
		# f.close()

		# Ranking images 
		ranked_images  = []
		for image_class in predicted_classes:
			ranked_images += [image_class+"_"+str(f) for f in os.listdir(os.path.join(image_path, image_class))]

		with open(result_folder + image_names[counter], 'w') as f:
			for item in ranked_images:
				f.write("%s\n" % item)
		f.close()

			




if __name__ == '__main__':

	categorize( )