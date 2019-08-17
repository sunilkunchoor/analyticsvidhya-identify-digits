#coding: utf-8

"""Identify the Digits
This problem is taken from https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-digits/. """

# Importing libraries
import pandas as pd
import os
import numpy as np


# Loading dataset
dataset = pd.read_csv('data/train.csv')
X = dataset.iloc[:,0:1]
Y = dataset.iloc[:,1:2]

# Path of train and test folder
TRAIN_PATH = 'data/images/train/' 
TEST_PATH = 'data/images/test/'

# Create training data set
X_train = X[:40000]
Y_train = Y[:40000]

# Create cross-validation set
X_dev = X[40000:42000]
Y_dev = Y[40000:42000]

# Plotting the numbers to check.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
for n in range(1,10):
    plt.subplot(1,10,n)
    img=mpimg.imread(TRAIN_PATH+X_train.iloc[n][0])
    plt.imshow(img,cmap='gray')
    plt.title(Y_train.iloc[n][0])

print(TRAIN_PATH+X_train.filename.values[1])

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of cross-validation examples = " + str(X_dev.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_dev shape: " + str(X_dev.shape))
print ("Y_dev shape: " + str(Y_dev.shape))

# defining a function to read images
from keras.preprocessing import image
from PIL import Image
def read_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224, 3))
    img = image.img_to_array(img)
    return img

# reading the images
import cv2
train_img = []
for img_path in X_train.filename.values:
    train_img.append(read_img(TRAIN_PATH+img_path))

# X_train_array = np.array(train_img, np.float16)
X_train_array1 = np.array(train_img)



# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D #Change to Conv2D if doesn't work
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = "relu"))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding the second Convolution layer and max pooling
classifier.add(Convolution2D(32, 3, 3, activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# Adding the full connection layer
classifier.add(Dense(output_dim = 128, activation = "relu"))
# Adding output layer
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


batch_size = 128
num_epoch = 10
#model training
classifier = classifier.fit(X_train_array1, Y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(X_dev, Y_dev))
