import os
import csv
import cv2
import tqdm
import scipy
import random
import numpy as np
import pickle as pkl
from read import load_images, load_test_images
from scipy.misc import imread
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torchsummary import summary


# Feature extractor
# ==============================================

def choose_alg(algo):
    if algo == 'kaze':
        alg = cv2.KAZE_create()
    elif algo == 'sift':
        alg = cv2.xfeatures2d.SIFT_create()
    elif algo == 'orb':
        alg = cv2.ORB_create()
    elif algo == 'surf':
        alg = cv2.xfeatures2d.SURF_create()
    return alg

# ==============================================
class Image_top_k_desc_features:

    def top_k_des(self, alg, image, vector_size=32):
        # image = imread(image_path, mode="RGB")
        try:       
            # Dinding image keypoints
            kps = alg.detect(image)

            # Getting first k of them. 
            # Number of keypoints is varies depend on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

            # computing descriptors vector
            kps, dsc = alg.compute(image, kps)
            desc_size = dsc.shape[1]
            # print desc_size

            # Flatten all of them in one big vector - our feature vector
            dsc = dsc.flatten()

            # Making descriptor of same size
            # Descriptor vector size is 64
            needed_size = (vector_size * desc_size)

            if dsc.size < needed_size:
                # if we have less the 32 descriptors then just adding zeros at the
                # end of our feature vector
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        
        except cv2.error as e:
            print 'Error: ', e
            return None

        return dsc
    

    def extract_features(self, alg, image):
        return self.top_k_des(alg, image)

    
    def batch_extractor(self, algo, images, labels, pickled_db_path="features.pkl"):
        alg = choose_alg(algo)
        result = {}

        for class_type_index in range(0, len(images)):
            class_type = images[class_type_index]
            class_type_label = labels[class_type_index]
            for img_index in range(0, len(class_type)):
                img = class_type[img_index]
                img_name = class_type_label[img_index]
                print 'Extracting features from image %s' % img_name
                result[img_name] = self.extract_features(alg, img)

        print result
        
        # saving all our feature vectors in pickled file
        with open(pickled_db_path, 'w') as fp:
            pkl.dump(result, fp)


# ==============================================
class Image_histogram_features:

    def __init__(self, images, labels, algo):
        self.images = images
        self.labels = labels
        self.algo = algo
        self.num_clusters = 70

    '''
    Load images, extract all feature descriptors and store them
    '''
    def extract_all_descriptors(self):
        alg = choose_alg(self.algo)
        features = []

        for class_type in self.images:
            for img in class_type:
                kps = alg.detect(img)
                kps, des = alg.compute(img, kps)
                # kp, des = sift.detectAndCompute(img,None)
                features = features + des.tolist()
        print "features extraction done."
        features = np.reshape(np.array(features), (-1,128))

        with open('other/'+self.algo+"_descriptors.pkl", 'w') as f:
            pickle.dump(features, f)


    '''
    Cluster those feature descriptors using MiniBatchKMeans
    and save the kmeans model
    '''
    def clustering(self):
        with open('other/'+self.algo+"_features.pkl", 'r') as f:
            features = pickle.load(f)

        print np.array(features).shape
        features = np.array(features)

        print "Clustering..." 
        kmeans = MiniBatchKMeans(n_clusters= self.num_clusters)
        kmeans.fit(features)

        with open('other/'+self.algo+'_kmeans_model.pkl', 'w') as f:
            pickle.dump(kmeans, f)


    '''
    Use the K means model to get Num-desc x 1 sized image feature
    which is basically the bin for each descriptor 
    '''
    def get_image_bag_data(self):
        with open('other/'+self.algo+'_kmeans_model.pkl', 'r') as f:
            kmeans = pickle.load(f)
        print 'model loaded'

        print "images Loaded"
        sift = cv2.xfeatures2d.SIFT_create()

        image_descs = []
        image_names = []

        print "bagging..."
        for class_type in range(len(self.images)):
            for img in range(len(self.images[class_type])):
                kp, des = sift.detectAndCompute(self.images[class_type][img],None)
                bag = kmeans.predict(des)
                image_descs.append(bag)
                image_names.append(self.labels[class_type][img])

        image_data = (image_descs, image_names)

        with open('other/'+self.algo+'_image_bag_data.pkl', 'w') as f:
            pickle.dump(image_data, f)


    '''
    Bin together the cluster counts to get a num_labels x 1 sized 
    image feature vector
    '''
    def get_image_hist_data(self):
        with open('other/'+self.algo+'_image_bag_data.pkl', 'r') as f:
            image_descs, image_names = pickle.load(f)

        hist_list = []
        # print image_descs

        print "Making histograms.."
        for i in range(len(image_descs)):
            hist = np.zeros(num_labels)
            num_desc = len(image_descs[i])
            for k in image_descs[i]:
                hist[k] += 1.0/num_desc
            hist_list.append(hist)

        image_data = (hist_list, image_names)

        result = {}
        for index in range(0, len(image_names)):
            result[image_names[index]] = hist_list[index]
        # print image_data

        with open('features/'+self.algo+'_features_hist_'+str(num_clusters)+'.pkl', 'w') as f:
            pickle.dump(result, f)


# ========================================================
class Image_VGG_features:

    def __init__(self, dataset_path, labels):
        self.labels = labels
        self.dataset_path = dataset_path
        self.model = VGG16(weights='imagenet', include_top=False)
        print self.model.summary()


    def get_feature(self, image_path):
        # load an image from file
        image = load_img(image_path, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # predict the probability across all output classes
        vgg_feature = self.model.predict(image)
        vgg_feature = np.array(vgg_feature).flatten()
        return vgg_feature

    def extract_features(self):
        result = {}
        for class_type in self.labels:
            for img_name in class_type:
                image_path = os.path.join(self.dataset_path, img_name)
                result[img_name] = self.get_feature(image_path)

        with open("vgg_features.pkl", "w") as f:
            pkl.dump(result, f)

# ========================================================

class Image_resnet18_features:

    def __init__(self, dataset_path, labels):
        self.labels = labels
        self.dataset_path = dataset_path
        self.model = models.resnet18(pretrained=True)
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')
        # Set model to evaluation mode
        self.model.eval()
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        summary(self.model, (3,224,224), device='cpu')


    def get_feature(self, image_name):
        # 1. Load the image with Pillow library
        img = Image.open(image_name)
        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(512)
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        self.model(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return my_embedding

    def extract_features(self):
        result = {}
        i=0
        for class_type in self.labels:
            for img_name in class_type:
                image_path = os.path.join(self.dataset_path, img_name)
                result[img_name] = np.array(self.get_feature(image_path))
                # print result[img_name].shape
                print i
                i+=1
        
        # print result
        with open("asgn2_19test_resnet18_features.pkl", "w") as f:
            pkl.dump(result, f)

# ========================================================

class Image_squeezenet_features:

    def __init__(self, dataset_path, labels):
        self.labels = labels
        self.dataset_path = dataset_path
        self.model = models.squeezenet1_0(pretrained=True)
        # Set model to evaluation mode
        self.model.eval()
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.pooling = nn.AvgPool2d(13)
        # self.model.cuda()
        summary(self.model, input_size=(3,224,224), device='cpu')


    def get_feature(self, image_name):
        # 1. Load the image with Pillow library
        img = Image.open(image_name)
        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)).cuda()
        
        h = self.pooling(self.model.features(t_img)).detach().cpu()
        # h = np.zeros(512)
        feature = np.array(h).flatten()
        # 8. Return the feature vector
        return feature

    def extract_features(self):
        result = {}
        i=0
        for class_type in self.labels:
            for img_name in class_type:
                image_path = os.path.join(self.dataset_path, img_name)
                result[img_name] = np.array(self.get_feature(image_path))
                # print result[img_name].shape
                print i
                i+=1
        
        # print result
        with open("squeezenet_features.pkl", "w") as f:
            pkl.dump(result, f)

# ========================================================

def main():

    # mode = 'train'
    mode = 'test'

    train_data_path = "../dataset"
    test_data_path = "../19"
    if mode == 'train':
        images, labels = load_images(dataset_path)
        dataset_path = train_data_path
    elif mode == 'test':
        images, labels = load_test_images(test_data_path)
        dataset_path = test_data_path
    else:
        print "Incorrect mode. Exiting!"
        exit(0)
    print "Images Loaded."

    print [len(i) for i in labels]
    
    '''
    top_k_features for ['sift', 'surf', 'orb', 'kaze']
    '''
    # top_k = Image_top_k_desc_features()
    # top_k.batch_extractor('surf', images, labels, 'surf_features_top_32.pkl')

    '''
    image histogram features for ['sift', 'surf', 'orb', 'kaze']]
    '''
    # hist = Image_histogram_features(images, labels, 'sift')
    # hist.extract_all_descriptors()
    # hist.clustering()
    # hist.get_image_bag_data()
    # hist.get_image_hist_data()

    '''
    VGG features
    '''
    # vgg = Image_VGG_features(dataset_path, labels)
    # vgg.extract_features()]

    '''
    ResNet18 features
    '''
    resnet18 = Image_resnet18_features(dataset_path, labels)
    resnet18.extract_features()

    '''
    SqueezeNet features
    '''
    # squeezenet = Image_squeezenet_features(dataset_path, labels)
    # squeezenet.extract_features()

# ========================================================

if __name__ == "__main__":
    main()
