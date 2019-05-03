import os, sys
import numpy as np 
import cv2

def load_images(input_path):
	image_data = {}
	class_folders = [os.path.join(input_path, f) for f in os.listdir(input_path)]
	# image_folders = [str(f) for f in os.listdir(input_path)]
	# images_list = [0]*len(image_class_folders)
	counter = 0;
	images_list = []

	for image_class in class_folders:
		sub_class_folders = [os.path.join(image_class, sub_class) for sub_class in os.listdir(image_class)]
		for sub_class in sub_class_folders:
			images_list.append([os.path.join(sub_class,f) for f in os.listdir(sub_class)])
			# image_labels[counter] = [os.path.join(image_folders[counter], f) for f in os.listdir(image_class)]
			counter += 1;

	if input_path[-1] != '/':
		input_path = input_path + '/'

	image_labels = [0]*counter
	counter = 0
	for image_sub_class_list in images_list:
		image_labels[counter] = [image_name.replace(input_path,"") for image_name in image_sub_class_list]
		counter += 1

	Images = [0]*counter

	counter = 0;
	# print image_labels

	for image_sub_class_list in images_list:
		Images[counter] = [cv2.imread(image, 0) for image in image_sub_class_list]
		counter += 1

	# print image_labels
	# for image_sub_class_list in image_labels:
	# 	for image_label in image_sub_class_list:
	# 		print image_label
	# 	print ""

	return Images, image_labels


def load_test_images(input_path):

	image_class_folders = [input_path]
	# image_folders = [str(f) for f in os.listdir(input_path)]
	images_list = [0]*len(image_class_folders)
	counter = 0;
	image_labels = [0]*len(image_class_folders)

	for image_class in image_class_folders:
		images_list[counter] = [os.path.join(image_class,f) for f in os.listdir(image_class)]
		# image_labels[counter] = [os.path.join(image_folders[counter], f) for f in os.listdir(image_class)]
		counter += 1;

	if input_path[-1] != '/':
		input_path = input_path + '/'

	counter = 0
	for image_class in images_list:
		image_labels[counter] = [image_name.replace(input_path,"") for image_name in image_class]
		counter += 1

	Images = [0]*len(image_class_folders);

	counter = 0
	
	for image_class in images_list:
		Images[counter] = [cv2.imread(image, 0) for image in image_class]
		counter += 1

	return Images, image_labels


if __name__ == '__main__':
	input_path = str(sys.argv[1])
	images, labels = load_images(input_path)





