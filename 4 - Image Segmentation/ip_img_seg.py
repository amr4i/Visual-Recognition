import numpy as np 
import os, sys
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy import ndimage
from tqdm import tqdm
from sklearn.cluster import KMeans


def read_images(path):
	img_files = [os.path.join(path,f) for f in os.listdir(path)]
	return img_files

def thresholding(img,fname):
	imgr = img.reshape(img.shape[0]* img.shape[1])
	for i in range(imgr.shape[0]):
		if imgr[i] > imgr.mean():
			imgr[i] = 3
		elif imgr[i] > 0.5:
			imgr[i] = 2
		elif imgr[i] > 0.25:
			imgr[i] = 1
		else:
			imgr[i] = 0

	threshImg = imgr.reshape(img.shape[0], img.shape[1])

	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(threshImg)
	fig.savefig("thresholding/" + fname.split(".")[0] + ".png")

def clustering(img, fname):

	colours = [[0,0,255], [0,255,0], [255,0,0], [0,0,0], [255,255,255]]
	img = np.array(img)/255.0
	nImg = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
	kmeans = KMeans(n_clusters=5).fit(nImg)
	clust = np.array([colours[i] for i in kmeans.labels_])
	clustImg = cv2.resize(clust.reshape(img.shape[0], img.shape[1], img.shape[2]), (img.shape[1], img.shape[0]))
	clustImg = clustImg*255.0
	cv2.imwrite("clustering/" + fname.split(".")[0] + ".png", clustImg)


def main():
	print('a')
	img_files = read_images('./sample/JPEGImages')
	print(img_files)
	for imf in tqdm(img_files):
		imgplt = plt.imread(imf)

		img = cv2.imread(imf)

		grImg = rgb2gray(imgplt)
		thresholding(grImg, imf.split('/')[-1])
		clustering(img, imf.split('/')[-1])

print('s')
if __name__ == '__main__':
	main()