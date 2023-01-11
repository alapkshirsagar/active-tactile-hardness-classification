
from imutils import paths
from tensorflow.keras import layers, Model, utils, applications, Sequential

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from os.path import join
import json


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


IMG_SIZE_0 = 240
IMG_SIZE_1 = 320
IMG_CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 200
NUM_FEATURES = 2048
SEQ_LENGTH = 5
NUM_CLASSES = 5



def create_model():
    cnn = Sequential()
    cnn.add(layers.Conv2D(6, 5, activation = "relu", input_shape = (IMG_SIZE_0, IMG_SIZE_1, IMG_CHANNELS)))
    cnn.add(layers.AveragePooling2D())
    cnn.add(layers.Conv2D(16,5,activation = "relu"))
    cnn.add(layers.AveragePooling2D())
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(units = 120, activation = "relu"))
    cnn.add(layers.Dense(units = 84 , activation = "relu"))

    
    # LSTM
    inp = layers.Input((None,IMG_SIZE_0, IMG_SIZE_1, IMG_CHANNELS))
    x = layers.TimeDistributed(cnn)(inp)
    x = layers.LSTM(32)(x)
    out = layers.Dense(NUM_CLASSES, activation = "softmax")(x)

    model = Model(inp, out)

    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
    model.summary()

    return model
    


model = create_model()

X = np.load("X.npy")
y = np.load("y.npy")

encoder = LabelBinarizer()
y = encoder.fit_transform(y)

print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 42)

print(y_train.shape)
history = model.fit(
    X_train,
    y_train,
    batch_size = BATCH_SIZE,
    epochs = 50,
)

results = model.evaluate(x_test, y_test, batch_size=128)
print(results)

