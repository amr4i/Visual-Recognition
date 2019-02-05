# import cv2
# import numpy as np

# print cv2.__version__
# img = cv2.imread('../train/detergent/N1_0.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)

# # img=cv2.drawKeypoints(gray,kp)

# # cv2.imwrite('sift_keypoints.jpg',img)


# img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg',img)

import os
import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt
from read import load_images


MIN_MATCH_COUNT = 10 
matched = {}
	
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

'''
Load images, extract all feature descriptors and store them
'''
def extract_SIFT_features():
	global features
	global img_names
	images, labels = load_images("../train")
	print "Loaded"
	
	features = []
	img_names = []

	for class_type_id in range(len(images)):
		class_type = images[class_type_id]
		for img_id in range(len(class_type)):
			img = class_type[img_id]
			
			# find the keypoints and descriptors with SIFT
			kp, des = sift.detectAndCompute(img,None)
			features.append([kp, des])
			img_names.append(labels[class_type_id][img_id])
	
	print "features done."

	# with open("image_SIFT_features.pkl", 'w') as f:
	# 	pickle.dump((features, img_names), f)

# ================================================================


def get_img_matches():
	global features
	global img_names

	# with open("image_SIFT_features.pkl", "r") as f:
	# 	features, img_names = pickle.load(f)

	for imgName in os.listdir("../sample_test/"):
		if(imgName.split(".")[1] != "jpg")
			continue

		print imgName
		img1 = cv2.imread("../sample_test/"+imgName)
		kp1, des1 = sift.detectAndCompute(img1,None)

		# FlANN based Matcher
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		 
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		good_classes = {}
		# adding all training images to flann matcher
		for imgId in range(len(features)):
			kp2, des2 = features[imgId]

			matches = flann.knnMatch(des1, des2, k=2)

			# store all the good matches as per Lowe's ratio test.
			good = []
			for m,n in matches:
			    if m.distance < 0.7*n.distance:
			        good.append(m)

			if len(good)>MIN_MATCH_COUNT:
				if img_names[imgId].split('/')[0] not in good_classes:
					good_classes[img_names[imgId].split('/')[0]] = 0

				good_classes[img_names[imgId].split('/')[0]] += 1

				# to show connecting descriptors in images
			    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			    # matchesMask = mask.ravel().tolist()

			    # # h,w,d = img1.shape
			    # h,w = img1.shape
			    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			    # dst = cv2.perspectiveTransform(pts,M)

			    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
			    # matched[imgName] = len(good)
			    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   #                 singlePointColor = None,
				   #                 matchesMask = matchesMask, # draw only inliers
				   #                 flags = 2)
			    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
			    # plt.imshow(img3, 'gray'),plt.show()

			
			# else:
			#     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
			#     matchesMask = None

		print good_classes
		predicted_classes = list(reversed(sorted(good_classes, key=good_classes.get)))

		with open("vanilla_sift_res/classes/"+imgName.split(".")[0]+".txt", "w") as f:
			for j in predicted_classes:
				f.write(j+"\n" )

		ranked_images  = []
		for image_class in predicted_classes:
			ranked_images += [image_class+"_"+str(f) for f in os.listdir(os.path.join("../train/", image_class))]

		with open("vanilla_sift_res/images/"+imgName.split(".")[0]+".txt", "w") as f:
			for item in ranked_images:
				f.write("%s\n" % item)


def main():
	extract_SIFT_features()
	get_img_matches()


if __name__ == "__main__":
	main()

