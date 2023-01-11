
from imutils import paths
from tensorflow.keras import  Model, utils, applications, Sequential
from tensorflow.keras.layers import *

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


def get_ConvLSTM(input_shape, num_classes): # input_shape = (SEQ_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    model = Sequential()

    # First Layer
    model.add(ConvLSTM2D(filters = 4, kernel_size = (3,3), activation = "tanh",
        data_format = "channels_last", recurrent_dropout = 0.2, return_sequences = True, input_shape = input_shape))
    model.add(MaxPooling3D( pool_size = (1,2,2),padding = "same", data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.2)))

    # Second Layer
    model.add(ConvLSTM2D(filters = 8, kernel_size = (3,3), activation = "tanh",
        data_format = "channels_last", recurrent_dropout = 0.2, return_sequences = True))
    model.add(MaxPooling3D( pool_size = (1,2,2),padding = "same", data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.2)))

    #Third Layer
    model.add(ConvLSTM2D(filters = 14, kernel_size = (3,3), activation = "tanh",
        data_format = "channels_last", recurrent_dropout = 0.2, return_sequences = True))
    model.add(MaxPooling3D( pool_size = (1,2,2),padding = "same", data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.2)))

    #Forth Layer
    model.add(ConvLSTM2D(filters = 16, kernel_size = (3,3), activation = "tanh",
        data_format = "channels_last", recurrent_dropout = 0.2, return_sequences = True))
    model.add(MaxPooling3D( pool_size = (1,2,2),padding = "same", data_format = "channels_last"))
    #model.add(TimeDistributed(Dropout(0.2)))


    model.add(Flatten())
    model.add(Dense(num_classes, activation = "softmax"))


    #model.summary()

    return model



if __name__ == "__main__":
    model = get_ConvLSTM((5,64,64,3), 5)

    
