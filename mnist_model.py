# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:36:02 2019

@author: Shantanu
"""

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt

image_index = 7777
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')

x_train.shape


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model = keras.Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)



model.save("mnist.h5")

model2 = keras.models.load_model("mnist.h5")

model2.evaluate(x_test, y_test)

import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps 
import numpy as np

img=Image.open('digit.png').convert('L')
img.thumbnail((28, 28), Image.ANTIALIAS)
img1 = PIL.ImageOps.invert(img)
img2 =  np.array(img1)

plt.imshow(img2, cmap=plt.cm.binary, interpolation="nearest")
plt.show()

print(model2.predict_classes(img2.reshape(1, 28, 28, 1))[0])
