import os,sys
import numpy as np 
import cv2
import scipy
import pickle as pkl 
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# device = torch.device('cuda:0')

# def batch_iter(data, batch_size, num_epochs, shuffle=False):
#     """Iterate the data batch by batch"""
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int(data_size / batch_size) + 1

#     for epoch in range(num_epochs):
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data

#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]

# class Net(nn.Module):

# 	def __init__(self, d_in, d_out, hid_dim=36, batch_size=64, num_epochs=200):
# 		super(Net, self).__init__()
# 		print d_in, d_out
# 		self.N = batch_size
# 		self.D_in = d_in
# 		self.D_out = d_out
# 		self.H = hid_dim
# 		self.model = nn.Sequential(
# 			nn.Linear(self.D_in, self.H),
# 			nn.ReLU(),
# 			nn.Linear(self.H, self.D_out),
# 			nn.Softmax(),
# 		).to(device)
# 		self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
# 		self.learning_rate = 1e-2
# 		self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
# 		self.batch_size = batch_size
# 		self.num_epochs = num_epochs

# 	def forward(self, x):
# 		x = self.model(x)
# 		return x



# def train_fine_model(X,y, PATH="resnet_squeezenet_fine_bilinearcnn_classifier_256.model"):
# 	X_train, X_test, y_train, y_test = train_test_split( X, y, 
# 		test_size=0.2, random_state=42, shuffle=False)

# 	iter_num = 1
# 	batch_size = 200
# 	num_epochs = 50
# 	net = Net(d_in=len(X_train[0]), d_out=len(y_train[0]), batch_size=batch_size, num_epochs=num_epochs)
# 	net.cuda()
# 	train_batches = batch_iter(list(zip(X_train, y_train)), batch_size, num_epochs)
# 	for train_batch in train_batches:
# 		print "Iteration number: "+str(iter_num),
# 		x, y = zip(*train_batch)
# 		# print len(x), len(y)
# 		x = Variable(torch.FloatTensor(list(x))).cuda()
# 		y = Variable(torch.LongTensor(list(y))).cuda()
# 		net.optimizer.zero_grad()
# 		y_pred = net(x)
# 		loss = net.loss_fn(y_pred, torch.max(y, 1)[1])
# 		loss.backward()
# 		net.optimizer.step()
# 		print ", Loss: " + str(loss.item())
# 		iter_num+=1
# 	print('Finished Training')

# 	correct = 0
# 	total = 0
# 	test_batches = batch_iter(list(zip(X_test, y_test)), self.batch_size, self.num_epochs)
# 	for data in test_batches:
# 	    x, labels = zip(*data)
# 	    outputs = net(Variable(x))
# 	    _, predicted = torch.max(outputs.data, 1)   # Find the class index with the maximum value.
# 	    total += labels.size(0)
# 	    correct += (predicted == labels).sum()

# 	print('Accuracy: %d %%' % (100 * correct / total))

# 	torch.save(model.state_dict(), PATH)


def multiply_vector(d1, d2):
	l = list(d1.keys())
	feature_vector = []
	for i in l:
		vec = np.outer(d1[i],d2[i]).flatten()
		feature_vector.append(vec)
	return np.array(feature_vector), l

		

def train_fine_model(X,y):
	X_train, X_test, y_train, y_test = train_test_split( X, y, 
		test_size=0.2, random_state=42, shuffle=False)



	classifier =  MLPClassifier(hidden_layer_sizes = (200,), verbose=True, 
		tol=0.0001, max_iter=40, batch_size=500)
	classifier.fit(X_train, y_train)
	acc = classifier.score(X_test,y_test)
	print "acc = %f" % acc

	with open("resnet_squeezenet_fine_bilinearcnn_classifier_36_40iter.pkl", "w") as f:
		pkl.dump(classifier, f)


def save_encoders(lec, lef, ohec, ohef):
	pkl.dump(lec, open('lec.pkl', 'w'))
	pkl.dump(lef, open('lef.pkl', 'w'))
	pkl.dump(ohec, open('ohec.pkl', 'w'))
	pkl.dump(ohef, open('ohef.pkl', 'w'))


def read_encoders():
	lec = pkl.load(open('coarse_encodings/lec.pkl', 'r'))
	lef = pkl.load(open('coarse_encodings/lef.pkl', 'r'))
	ohec = pkl.load(open('coarse_encodings/ohec.pkl', 'r'))
	ohef = pkl.load(open('coarse_encodings/ohef.pkl', 'r'))
	return lec,lef,ohec,ohef


def one_hot_labels(image_labels, train_encoding=True):
	N = len(image_labels)
	coarse_labels = []
	fine_labels = []
	image_name = []
	for image_label in image_labels:
		image_name.append(image_label.split("/")[2])
		fine_labels.append(image_label.split("/")[0]+'@'+image_label.split("/")[1])
		coarse_labels.append(image_label.split("/")[0])

	if train_encoding == True:
		lec = LabelEncoder()
		lef = LabelEncoder()
		ohec = OneHotEncoder(categorical_features = [0])
		ohef = OneHotEncoder(categorical_features = [0])
		labels_coarse = lec.fit_transform(coarse_labels).reshape((N,1))
		labels_fine = lef.fit_transform(fine_labels).reshape((N,1))
		ohe_coarse_labels = ohec.fit_transform(labels_coarse).toarray()
		ohe_fine_labels = ohef.fit_transform(labels_fine).toarray()
		save_encoders(lec,lef,ohec,ohef)
	else:
		lec,lef,ohec,ohef = read_encoders()
		labels_coarse = lec.transform(coarse_labels).reshape((N,1))
		labels_fine = lef.transform(fine_labels).reshape((N,1))
		ohe_coarse_labels = ohec.transform(labels_coarse).toarray()
		ohe_fine_labels = ohef.transform(labels_fine).toarray()
	
	return ohe_coarse_labels, ohe_fine_labels


def main():

	feature_dictionary_1 = pkl.load(open("features/resnet18_features.pkl", "r"))
	feature_dictionary_2 = pkl.load(open("features/squeezenet_features.pkl", "r"))

	feature_vector, image_labels = multiply_vector(feature_dictionary_1, feature_dictionary_2)
	# feature_vector = np.array(feature_dictionary.values())
	print feature_vector.shape
	zipped = list(zip(image_labels, feature_vector))
	random.shuffle(zipped)
	image_labels, feature_vector = zip(*zipped)

	ohe_coarse_labels, ohe_fine_labels =  one_hot_labels(image_labels, True)
	print "Training...."
	train_fine_model(feature_vector, ohe_fine_labels)
	


if __name__ == '__main__':
	main()
	
