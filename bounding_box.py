import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import os,sys

path = '/home/aarsh/dataset_train/train/3m_high_tack_spray_adhesive/N2_90.jpg'

fig = plt.figure()

xmin = 270
ymin = 70
xmax = 355
ymax = 230
ax = fig.add_axes([0,0,1,1])
image = plt.imread(path)
plt.imshow(image)

width = xmax - xmin;
height = ymax - ymin

edgecolor = 'r';

rect  = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
ax.add_patch(rect)

plt.show()
