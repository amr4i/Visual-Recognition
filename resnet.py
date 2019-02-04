import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2
import os
from read import load_images
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint



# Fixed for our Cats & Dogs classes
NUM_CLASSES = 16
# Fixed for Cats & Dogs color images
CHANNELS = 3

IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 96
BATCH_SIZE_VALIDATION = 96

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

resnet_weights_path = '../resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


#Still not talking about our train/test data or any pre-processing.

model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = resnet_weights_path))

# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

model.summary()


sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

image_size = IMAGE_RESIZE

data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)


train_generator = data_generator.flow_from_directory(
        '../train',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '../valid',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical') 

# Max number of steps that these generator will have opportunity to process their source content
# len(train_generator) should be 'no. of available train images / BATCH_SIZE_TRAINING'
# len(valid_generator) should be 'no. of available train images / BATCH_SIZE_VALIDATION'
# (BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(validation_generator))

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = '../working/resnet_model.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

# Pseudo code for hyperparameters Grid Search

'''
from sklearn.grid_search import ParameterGrid
param_grid = {'epochs': [5, 10, 15], 'steps_per_epoch' : [10, 20, 50]}

grid = ParameterGrid(param_grid)

# Accumulate history of all permutations (may be for viewing trend) and keep watching for lowest val_loss as final model
for params in grid:
    print(params)
'''

fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_checkpointer, cb_early_stopper]
)

model_json = model.to_json()
with open("model_resnet.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_resnet.h5")

print("Saved model to disk")

# model.load_weights("../working/resnet_model.hdf5")

print(fit_history.history.keys())

# plt.figure(1, figsize = (15,8)) 
    
# plt.subplot(221)  
# plt.plot(fit_history.history['acc'])  
# plt.plot(fit_history.history['val_acc'])  
# plt.title('model accuracy')  
# plt.ylabel('accuracy')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'valid']) 
    
# plt.subplot(222)  
# plt.plot(fit_history.history['loss'])  
# plt.plot(fit_history.history['val_loss'])  
# plt.title('model loss')  
# plt.ylabel('loss')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'valid']) 

# plt.show()

# # NOTE that flow_from_directory treats each sub-folder as a class which works fine for training data
# # Actually class_mode=None is a kind of workaround for test data which too must be kept in a subfolder

# # batch_size can be 1 or any factor of test dataset size to ensure that test dataset is samples just once, i.e., no data is left out
# test_generator = data_generator.flow_from_directory(
#     directory = '../sample_test',
#     target_size = (image_size, image_size),
#     batch_size = BATCH_SIZE_TESTING,
#     class_mode = None,
#     shuffle = False,
#     seed = 123
# )

# # Try batch size of 1+ in test_generator & check batch_index & filenames in resulting batches
# '''
# for i in test_generator:
#     #print(test_generator.batch_index, test_generator.batch_size)
#     idx = (test_generator.batch_index - 1) * test_generator.batch_size
#     print(test_generator.filenames[idx : idx + test_generator.batch_size])
# '''


# test_generator.reset()

# pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

# predicted_class_indices = np.argmax(pred, axis = 1)


# TEST_DIR = '../sample_test'
# f, ax = plt.subplots(5, 5, figsize = (15, 15))

# for i in range(0,25):
#     imgBGR = cv2.imread(TEST_DIR + test_generator.filenames[i])
#     imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    
#     # a if condition else b
#     predicted_class = "Dog" if predicted_class_indices[i] else "Cat"

#     ax[i//5, i%5].imshow(imgRGB)
#     ax[i//5, i%5].axis('off')
#     ax[i//5, i%5].set_title("Predicted:{}".format(predicted_class))    

# plt.show()

# results_df = pd.DataFrame(
#     {
#         'id': pd.Series(test_generator.filenames), 
#         'label': pd.Series(predicted_class_indices)
#     })
# results_df['id'] = results_df.id.str.extract('(\d+)')
# results_df['id'] = pd.to_numeric(results_df['id'], errors = 'coerce')
# results_df.sort_values(by='id', inplace = True)

# results_df.to_csv('submission.csv', index=False)
# results_df.head()