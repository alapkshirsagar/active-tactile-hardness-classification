
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

import einops


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# =================================================== CONV LSTM ===========================================================

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


    model.summary()

    return model


# ================================================= Spatio Temporal Convolution ======================================================
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/video/video_classification.ipynb

class Conv2Plus1D(Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension. 
    """
    super().__init__()
    self.seq = Sequential([  
        # Spatial decomposition
        Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])
  
  def call(self, x):
    return self.seq(x)

class ResidualMain(Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        LayerNormalization(),
        ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        LayerNormalization()
    ])
    
  def call(self, x):
    return self.seq(x)

class Project(Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
  """
  def __init__(self, units):
    super().__init__()
    self.seq = Sequential([
        Dense(units),
        LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)



def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)
  
  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return add([res, out])



class ResizeVideo(Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  
      
      Args:
        video: Tensor representation of the video, in the form of a set of frames.
      
      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos


def get_spatio_temporal_Conv(input_shape, num_classes):
    _, HEIGHT, WIDTH, _=  input_shape

    input = Input(shape=(input_shape))
    x = input

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ResizeVideo(HEIGHT // 8, WIDTH // 4)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)
    x = Dense(num_classes)(x)

    model = Model(input, x)
    model.summary()
    return model

     

     

     




if __name__ == "__main__":
    model = get_ConvLSTM((5,64,64,3), 5)

    
