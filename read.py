import os, sys
import numpy as np 
import cv2

def load_images(input_path):

	image_class_folders = [os.path.join(input_path, f) for f in os.listdir(input_path)]
	# image_folders = [str(f) for f in os.listdir(input_path)]
	images_list = [0]*len(image_class_folders)
	counter = 0;
	image_labels = [0]*len(image_class_folders)

	for image_class in image_class_folders:
		images_list[counter] = [os.path.join(image_class,f) for f in os.listdir(image_class)]
		# image_labels[counter] = [os.path.join(image_folders[counter], f) for f in os.listdir(image_class)]
		counter += 1;

	counter = 0
	for image_class in images_list:
		image_labels[counter] = [image_name.replace(input_path+"/","") for image_name in image_class]
		counter += 1

	Images = [0]*len(image_class_folders);

	counter = 0;
	print image_labels

	for image_class in images_list:
		Images[counter] = [cv2.imread(image, 0) for image in image_class]
		counter += 1

	# for image_class in image_labels:
	# 	for image in image_class:
	# 		print image
	# 	print ""

	return Images, image_labels

if __name__ == '__main__':
	input_path = str(sys.argv[1])
	images, labels = load_images(input_path)





