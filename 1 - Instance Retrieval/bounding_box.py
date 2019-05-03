import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import os,sys

path = '/home/aarsh/dataset_train/train/vo5_split_ends_anti_breakage_shampoo/N1_0.jpg'

fig = plt.figure()

xmin = 280
ymin = 120
xmax = 340
ymax = 270
ax = fig.add_axes([0,0,1,1])
image = plt.imread(path)
plt.imshow(image)

width = xmax - xmin;
height = ymax - ymin

edgecolor = 'r';

rect  = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
ax.add_patch(rect)

plt.show()
