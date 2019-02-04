import os, sys
import numpy as np 

input_path = str(sys.argv[1])

image_class_folders = [os.path.join(input_path, f) for f in os.listdir(input_path)]
images_list = [0]*len(image_class_folders)
counter = 0;
for image_class in image_class_folders:
	images_list[counter] = [os.path.join(image_class,f) for f in os.listdir(image_class)]
	counter += 1;

print images_list



