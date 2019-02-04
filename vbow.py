import os
import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from read import load_images
import pickle
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


num_labels = 70
sift = cv2.xfeatures2d.SIFT_create()

'''
Load images, extract all feature descriptors and store them
'''
def extract_all_SIFT():
	images, labels = load_images("../train")
	print "Loaded"
	

	features = []

	for class_type in images:
		for img in class_type:
			kp, des = sift.detectAndCompute(img,None)
			features = features + des.tolist()
	print "features done."

	features = np.reshape(np.array(features), (-1,128))


	with open("features.pkl", 'w') as f:
		pickle.dump(features, f)

# ==============================================

'''
Cluster those feature descriptors using MiniBatchKMeans
and save the kmeans model
'''
def clustering():
	with open("features.pkl", 'r') as f:
		features = pickle.load(f)
	print "Loaded"


	print np.array(features).shape
	features = np.array(features)


	kmeans = MiniBatchKMeans(n_clusters= num_labels)
	kmeans.fit(features)

	with open('model.pkl', 'w') as f:
		pickle.dump(kmeans, f)

# ====================================================

'''
Use the K means model to get Num-desc x 1 sized image feature
which is basically the bin for each descriptor 
'''

def get_image_bag_data():
	with open('kmeans_model.pkl', 'r') as f:
		kmeans = pickle.load(f)
	print 'model loaded'

	images, labels = load_images("../train")
	print "images Loaded"
	sift = cv2.xfeatures2d.SIFT_create()

	image_descs = []
	image_names = []

	for class_type in range(len(images)):
		for img in range(len(images[class_type])):
			kp, des = sift.detectAndCompute(images[class_type][img],None)
			bag = kmeans.predict(des)
			image_descs.append(bag)
			image_names.append(labels[class_type][img])

	image_data = (image_descs, image_names)

	with open('image_bag_data.pkl', 'w') as f:
		pickle.dump(image_data, f)

# ========================================================

'''
Bin together the cluster counts to get a num_labels x 1 sized 
image feature vector
'''

def get_image_hist_data():
	with open('image_bag_data.pkl', 'r') as f:
		image_descs, image_names = pickle.load(f)

	hist_list = []
	# print image_descs

	for i in range(len(image_descs)):
		hist = np.zeros(num_labels)
		num_desc = len(image_descs[i])
		for k in image_descs[i]:
			hist[k] += 1.0/num_desc
		hist_list.append(hist)

	image_data = (hist_list, image_names)
	# print image_data

	with open('image_hist_data.pkl', 'w') as f:
		pickle.dump(image_data, f)

# ============================================================

def training():
	global classes
	with open('image_hist_data.pkl', 'r') as f:
		image_hist_list, image_names = pickle.load(f)

	X = np.array(image_hist_list)
	Y = []
	y_labels = [ i.split('/')[0] for i in image_names ]
	classes = list(set(y_labels))
	# print y_labels
	print classes
	
	for s in y_labels:
		Y.append(classes.index(s))



	X, Y = shuffle(X, Y, random_state=0)

	mlp = MLPClassifier(verbose=False, \
							solver='adam', \
							max_iter=500000, shuffle=True,
							n_iter_no_change=1000)
	mlp.fit(X, Y)

	# rfc = RandomForestClassifier(verbose=False, n_estimators=100)
	# rfc.fit(X,Y)

	with open('mlp_model_adam.pkl', 'w') as f:
		pickle.dump(mlp, f)


# ================================================================

def predict():
	global classes
	# classes = ['vo5_extra_body_volumizing_shampoo', \
	# 			'coca_cola_glass_bottle', \
	# 			'nutrigrain_apple_cinnamon', \
	# 			'palmolive_green', \
	# 			'detergent', \
	# 			'aunt_jemima_original_syrup', \
	# 			'pringles_bbq', \
	# 			'nice_honey_roasted_almonds', \
	# 			'expo_marker_red', \
	# 			'clif_crunch_chocolate_chip', \
	# 			'vo5_split_ends_anti_breakage_shampoo', \
	# 			'3m_high_tack_spray_adhesive', \
	# 			'cheez_it_white_cheddar', \
	# 			'listerine_green', \
	# 			'campbells_chicken_noodle_soup', \
	# 			'cholula_chipotle_hot_sauce']

	with open("mlp_model_adam.pkl", 'r') as f:
		mlp = pickle.load(f)

	with open('kmeans_model.pkl', 'r') as f:
		kmeans = pickle.load(f)

	for imageFile in os.listdir("../sample_test/"):
		if(imageFile.split(".")[1] != "jpg"):
			continue
		# print imageFile
		img = cv2.imread("../sample_test/"+imageFile)
		kp, des = sift.detectAndCompute(img, None)
		
		x = np.zeros(num_labels)
		nkp = np.size(kp)

		for d in des:
			idx = kmeans.predict([d])
			x[idx] += 1.0/nkp
		# print x

		res = mlp.predict_proba([x])
		print imageFile + ":\t "+ classes[np.argmax(res)]
		print str(res) + "\n"

# ===============================================================

def main():
	# extract_all_SIFT()
	# clustering()
	# get_image_bag_data()
	# get_image_hist_data()
	training()
	predict()

if __name__ == "__main__":
	main()