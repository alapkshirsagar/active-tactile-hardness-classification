from imutils import paths
from tensorflow.keras import Model, utils, applications, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework.ops import EagerTensor

from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from os.path import join
import json
import torch.nn as nn

import einops

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score



VERSION_FLAG = "windows"

class TorchHistory:
    def __init__(self):
        self.history = {}


# =================================================== CONV LSTM ===========================================================


class ConvLSTM:
    def __init__(self):
        self.model_name = "ConvLSTM"
        self.model = None
    def get_model(
        self, input_shape, num_classes
    ):  # input_shape = (SEQ_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
        model = Sequential()
        

        # First Layer
        model.add(
            ConvLSTM2D(
                filters=4,
                kernel_size=(3, 3),
                activation="tanh",
                data_format="channels_last",
                recurrent_dropout=0.2,
                return_sequences=True,
                input_shape=input_shape,
            )
        )
        model.add(
            MaxPooling3D(
                pool_size=(1, 2, 2), padding="same", data_format="channels_last"
            )
        )
        model.add(TimeDistributed(Dropout(0.2)))

        # Second Layer
        model.add(
            ConvLSTM2D(
                filters=8,
                kernel_size=(3, 3),
                activation="tanh",
                data_format="channels_last",
                recurrent_dropout=0.2,
                return_sequences=True,
            )
        )
        model.add(
            MaxPooling3D(
                pool_size=(1, 2, 2), padding="same", data_format="channels_last"
            )
        )
        model.add(TimeDistributed(Dropout(0.2)))

        # Third Layer
        model.add(
            ConvLSTM2D(
                filters=14,
                kernel_size=(3, 3),
                activation="tanh",
                data_format="channels_last",
                recurrent_dropout=0.2,
                return_sequences=True,
            )
        )
        model.add(
            MaxPooling3D(
                pool_size=(1, 2, 2), padding="same", data_format="channels_last"
            )
        )
        model.add(TimeDistributed(Dropout(0.2)))

        # Forth Layer
        model.add(
            ConvLSTM2D(
                filters=16,
                kernel_size=(3, 3),
                activation="tanh",
                data_format="channels_last",
                recurrent_dropout=0.2,
                return_sequences=True,
            )
        )
        model.add(
            MaxPooling3D(
                pool_size=(1, 2, 2), padding="same", data_format="channels_last"
            )
        )
        model.add(TimeDistributed(Dropout(0.2)))

        model.add(Flatten())
        model.add(Dense(num_classes, activation="softmax"))

        # model.summary()
        self.model = model

        return model

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return None

    def train_model(
        self,
        X_train,
        y_train,
        batch_size,
        epochs,
        validation_samples_per_class,
        train_samples_per_class,
    ):
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_samples_per_class
            / (train_samples_per_class + validation_samples_per_class),
        )
        return history

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def predict_raw(self, X, **kwargs):
        return self.model(X, **kwargs)

    def save_model(self, path):
        self.model.save(path)
        return None


class OpticalFlowNN:
    def compute_optical_flow(self, video):
        return compute_optical_flow(video, max_features=self.max_features)

    def __init__(self, max_features=500):
        self.model_name = "OpticalFlowNN"
        self.model = None
        self.max_features = 500

    def get_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation="softmax"))
        self.model = model
        return model

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return None

    def train_model(
        self,
        X_train,
        y_train,
        validation_samples_per_class=None,
        train_samples_per_class=None,
        epochs=20,
        batch_size=32,
    ):
        # Compute optical flow features for each video
        X_features_original = [self.compute_optical_flow(video) for video in X_train]
        X_features = np.array(X_features_original)

        # Convert y_train to integer labels if one-hot encoded
        y_train = np.argmax(y_train, axis=1)

        # Convert labels to categorical (one-hot encoding)
        y_train = to_categorical(y_train)

        validation_split = 0

        if validation_samples_per_class is None or validation_samples_per_class == 0:
            validation_split = None
        else:
            validation_split = validation_samples_per_class / (
                validation_samples_per_class + train_samples_per_class
            )

        # Train the model
        history = self.model.fit(
            X_features,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )
        return history


    def predict(self, X, **kwargs):
        X = [self.compute_optical_flow(video) for video in X]
        X = np.array(X)
        return self.model.predict(X, **kwargs)

    def predict_raw(self, X, **kwargs):
        X = [self.compute_optical_flow(np.array(video)) for video in X]
        X = np.array(X)
        return self.model(X, **kwargs)

    def save_model(self, path):
        self.model.save(path)
        return None

class DINOKeras:
    def __init__(self):
        self.model_name = "DINOKeras"
        self.model = None
        self.device = "cpu"

    def get_model(self, input_shape, num_classes):
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.patch_size = 14
        self.backbone.eval()
        
        model = Sequential()
        model.add(
            Dense(128, input_dim=self.backbone.num_features * input_shape[0], activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation="softmax"))
        self.model = model
        
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.patch_size = 14
        self.backbone.eval()
        return model

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return None

    def train_model(
        self,
        X_train,
        y_train,

        validation_samples_per_class,
        train_samples_per_class,
        batch_size = 32,
        epochs = 20,
    ):
        self.backbone = self.backbone.to(self.device)
        
        # Split the data into training and validation sets

        validation_split = 0
        
        if validation_samples_per_class is None or validation_samples_per_class == 0:
            validation_split = None
        else:
            validation_split = validation_samples_per_class / (
                validation_samples_per_class + train_samples_per_class
            )
        X_train = torch.tensor(X_train)
        X_train = X_train.to(self.device)
        X_train_embeddings = self.get_embeddings(X_train)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_train = y_train.to(self.device)
        X_train_embeddings = X_train_embeddings.cpu().numpy()
        y_train = y_train.cpu().numpy()
        
        history = self.model.fit(
            X_train_embeddings,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )
        return history
    
    def predict(self, X, **kwargs):
        X = torch.tensor(X)
        X = X.to(self.device)
        X_embeddings = self.get_embeddings(X)
        # convert to numpy array
        X_embeddings = X_embeddings.cpu().numpy()
        return self.model.predict(X_embeddings, **kwargs)
    
    def predict_raw(self, X, **kwargs):
        # X = torch.tensor(X)
        # X = X.to(self.device)
        X_embeddings = self.get_embeddings(X)
        X_embeddings = X_embeddings.cpu().numpy()

        return self.model(X_embeddings, **kwargs)
     
    def pad_to_multiple_of_patch_size(self, X, patch_size):
        _, height, width, _ = X.shape
        pad_height = (patch_size - height % patch_size) % patch_size
        pad_width = (patch_size - width % patch_size) % patch_size
        # Convert TensorFlow tensor to NumPy array
        # if X is a PyTorch tensor
        np_array = None
        
        if isinstance(X, torch.Tensor) or isinstance(X, EagerTensor):
            np_array = X.numpy()
        elif isinstance(X, np.ndarray):
            np_array = X
        else:
            raise ValueError("Input must be a NumPy array or PyTorch tensor")
        # Convert NumPy array to PyTorch tensor
        # torch_tensor = np_array
        # if VERSION_FLAG  != "windows":
        torch_tensor = torch.from_numpy(np_array)

        return torch.nn.functional.pad(torch_tensor, (0, 0, 0, pad_width, 0, pad_height))

    def get_embeddings(self, X):
        with torch.no_grad():
            # X = torch.tensor(X)
            embeddings = torch.tensor([])
            for i in range(X.shape[1]):
                # self.visualize_dino_features(X[:,i,:,:,:])
                X_resize = self.pad_to_multiple_of_patch_size(
                    X[:, i, :, :, :], patch_size=self.patch_size
                )
                X_normalize = X_resize / 255.0
                X_permute = X_normalize.permute(0, 3, 1, 2)
                embedding = self.backbone(X_permute.float())
                embeddings = torch.cat((embeddings, embedding), dim=1)
                # if i == 0:
                #     embeddings = embedding
                # else:
                #     embeddings = embeddings + embedding
        return embeddings
            
            
            


class Conv2Plus1D(Layer):
    def __init__(self, filters, kernel_size, padding):
        """
        A sequence of convolutional layers that first apply the convolution operation over the
        spatial dimensions, and then the temporal dimension.
        """
        super().__init__()
        self.seq = Sequential(
            [
                # Spatial decomposition
                Conv3D(
                    filters=filters,
                    kernel_size=(1, kernel_size[1], kernel_size[2]),
                    padding=padding,
                ),
                # Temporal decomposition
                Conv3D(
                    filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding
                ),
            ]
        )

    def call(self, x):
        return self.seq(x)


class ResidualMain(Layer):
    """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
    """

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = Sequential(
            [
                Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding="same"),
                LayerNormalization(),
                ReLU(),
                Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding="same"),
                LayerNormalization(),
            ]
        )

    def call(self, x):
        return self.seq(x)


class Project(Layer):
    """
    Project certain dimensions of the tensor as the data is passed through different
    sized filters and downsampled.
    """

    def __init__(self, units):
        super().__init__()
        self.seq = Sequential([Dense(units), LayerNormalization()])

    def call(self, x):
        return self.seq(x)


def add_residual_block(input, filters, kernel_size):
    """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters, kernel_size)(input)

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
        old_shape = einops.parse_shape(video, "b t h w c")
        images = einops.rearrange(video, "b t h w c -> (b t) h w c")
        images = self.resizing_layer(images)
        videos = einops.rearrange(images, "(b t) h w c -> b t h w c", t=old_shape["t"])
        return videos


def get_spatio_temporal_Conv(input_shape, num_classes):
    _, HEIGHT, WIDTH, _ = input_shape

    input = Input(shape=(input_shape))
    x = input

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding="same")(x)
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


def compute_optical_flow(video_frames, max_features=500):
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    flow_features = np.array([])

    for i in range(len(video_frames) - 1):
        prev_frame = video_frames[i].astype(
            np.uint8
        )  # Ensure the frame is in the correct format
        next_frame = video_frames[i + 1].astype(
            np.uint8
        )  # Ensure the frame is in the correct format

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Detect features to track
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        if p0 is None:
            continue

        # Calculate optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, p0, None, **lk_params
        )

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Calculate flow vectors
        flow = good_new - good_old
        flow_features = np.concatenate([flow_features, flow.flatten()])

    # Pad or truncate to max_features length
    if flow_features.shape[0] < max_features:
        flow_features = np.pad(
            flow_features, (0, max_features - len(flow_features)), "constant"
        )
    else:
        flow_features = flow_features[:max_features]

    return flow_features


if __name__ == "__main__":
    model = get_ConvLSTM((5, 64, 64, 3), 5)
