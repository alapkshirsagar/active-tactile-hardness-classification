from imutils import paths
from tensorflow.keras import Model, utils, applications, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

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


class DINOv2NN:
    def __init__(self):
        self.model_name = "DINOv2NN"
        self.model = None
        self.device = "cpu"

    def visualize_dino_features(self, X):
        patchsize = self.patch_size
        X_resize = self.pad_to_multiple_of_patch_size(X, patch_size=self.patch_size)
        X_normalize = X_resize / 255.0
        X_permute = X_normalize.permute(0, 3, 1, 2)
        total_features = []
        patch_h = X_resize.shape[1] // patchsize
        patch_w = X_resize.shape[2] // patchsize
        for image in X_permute:
            features_dict = self.backbone.forward_features(image.unsqueeze(0))
            features = features_dict["x_norm_patchtokens"]
            total_features.append(features)
        total_features = torch.cat(total_features, dim=0)
        total_features = total_features.reshape(
            X.shape[0] * patch_h * patch_w, self.backbone.num_features
        )
        pca = PCA(n_components=3)
        pca.fit(total_features)
        pca_features = pca.transform(total_features)
        # visualize PCA components for finding a proper threshold
        # 3 histograms for 3 components
        plt.subplot(2, 2, 1)
        plt.hist(pca_features[:, 0])
        plt.subplot(2, 2, 2)
        plt.hist(pca_features[:, 1])
        plt.subplot(2, 2, 3)
        plt.hist(pca_features[:, 2])
        plt.savefig("dino_features_histogram.png")
        # Clear the plot
        plt.clf()

        # min_max scale
        pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (
            pca_features[:, 0].max() - pca_features[:, 0].min()
        )
        for i in range(X_resize.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(
                pca_features[
                    i * patch_h * patch_w : (i + 1) * patch_h * patch_w, 0
                ].reshape(patch_h, patch_w)
            )
        plt.savefig("dino_features.png")
        # Clear the plot
        plt.clf()

        # segment/seperate the backgound and foreground using the first component
        pca_features_bg = pca_features[:, 0] > 0.35  # from first histogram
        pca_features_fg = ~pca_features_bg
        # plot the pca_features_bg
        for i in range(X_resize.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(
                pca_features_bg[
                    i * patch_h * patch_w : (i + 1) * patch_h * patch_w
                ].reshape(patch_h, patch_w)
            )
        plt.savefig("dino_features_bg.png")
        # Clear the plot
        plt.clf()

        # 2nd PCA for only foreground patches
        pca.fit(total_features[pca_features_fg])
        pca_features_left = pca.transform(total_features[pca_features_fg])
        for i in range(3):
            # min_max scaling
            pca_features_left[:, i] = (
                pca_features_left[:, i] - pca_features_left[:, i].min()
            ) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

        pca_features_rgb = pca_features.copy()
        # for black background
        pca_features_rgb[pca_features_bg] = 0
        # new scaled foreground features
        pca_features_rgb[pca_features_fg] = pca_features_left

        # reshaping to numpy image format
        pca_features_rgb = pca_features_rgb.reshape(
            X_resize.shape[0], patch_h, patch_w, 3
        )
        for i in range(X_resize.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(pca_features_rgb[i])
        plt.savefig("dino_features_fg.png")

    def get_model(self, input_shape, num_classes):
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.patch_size = 14
        self.backbone.eval()
        # Define the first linear layer
        linear1 = nn.Linear(self.backbone.num_features * input_shape[0], 128)
        # Add dropout layer
        dropout = nn.Dropout(0.2)
        relu1 = nn.ReLU()
        linear2 = nn.Linear(128, num_classes)
        softmax = nn.Softmax(dim=1)


        # Create the model
        self.model = nn.Sequential(linear1, dropout, relu1, linear2, softmax)
        return self.model

    def pad_to_multiple_of_patch_size(self, X, patch_size):
        _, height, width, _ = X.shape
        pad_height = (patch_size - height % patch_size) % patch_size
        pad_width = (patch_size - width % patch_size) % patch_size
        # Convert TensorFlow tensor to NumPy array
        np_array = X.numpy()

        # Convert NumPy array to PyTorch tensor
        torch_tensor = torch.tensor(np_array)

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

    def compile_model(self, loss, metrics, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = metrics
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
        train_accuracy = []
        val_accuracy = []
        train_loss=[]
        val_loss=[]
        # Move the model to the GPU
        self.model = self.model.to(self.device)
        # Move the backbone to the GPU
        self.backbone = self.backbone.to(self.device)
        
        validation_split = 0

        if validation_samples_per_class is None or validation_samples_per_class == 0:
            validation_split = 0
        else:
            validation_split = validation_samples_per_class / (
                validation_samples_per_class + train_samples_per_class
            )

        # Split the data into training and validation sets
        if(validation_split == 0):
            X_train, X_val, y_train, y_val = X_train, [], y_train, []
        else:
            X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_split,
            stratify=y_train,
        )
        if(validation_samples_per_class != 0):
            print("Train samples: ", X_train.shape[0], "Validation samples: ", X_val.shape[0])
        else :
            print("Train samples: ", X_train.shape[0])
        # Convert the data to PyTorch tensors and move to GPU
        X_train = torch.tensor(X_train)
        X_train = X_train.to(self.device)
        X_train_embeddings = self.get_embeddings(X_train)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_train = y_train.to(self.device)
        if validation_samples_per_class != 0:
            X_val = torch.tensor(X_val)
            X_val = X_val.to(self.device)
            X_val_embeddings = self.get_embeddings(X_val)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            y_val = y_val.to(self.device)
        # Create a DataLoader for the training and validation sets
        train_dataset = torch.utils.data.TensorDataset(X_train_embeddings, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        if validation_samples_per_class != 0:
            val_dataset = torch.utils.data.TensorDataset(X_val_embeddings, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        # Train the model
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0  # To keep track of the number of batches
            all_data = []
            all_labels = []
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.loss(y_pred, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                all_data.append(X_batch)
                all_labels.append(y_batch)
                batch_count += 1
            all_data = torch.cat(all_data)
            all_labels = torch.cat(all_labels)
            
            with torch.no_grad():
                # Get model predictions
                outputs = self.model(all_data)
                _, predicted = torch.max(outputs, 1)
                # convert from one-hot to class index
                all_labels = torch.argmax(all_labels, dim=1)
                # Calculate accuracy
                correct = (predicted == all_labels).sum().item()
                epoch_accuracy = correct / all_labels.size(0)

            epoch_loss = total_loss / batch_count  # Dividing total loss by the number of batches

            train_accuracy.append(epoch_accuracy)
            train_loss.append(epoch_loss)
            
            
            # Evaluate the model on the validation set per epoch
            self.model.eval()

            if validation_samples_per_class != 0:
                all_data = []
                all_labels = []
                total_loss = 0
                total_val = 0
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    _, predicted = torch.max(y_pred, 1)
                    total_val += X_batch.size(0)
                    
                    total_loss += self.loss(y_pred, y_batch).item()
                    all_data.append(X_batch)
                    all_labels.append(y_batch)
                    
                val_loss.append(total_loss / total_val)
                val_accuracy.append(epoch_accuracy)
                all_data = torch.cat(all_data)
                all_labels = torch.cat(all_labels)
                
                with torch.no_grad():
                    # Get model predictions
                    outputs = self.model(all_data)
                    _, predicted = torch.max(outputs, 1)
                    # convert from one-hot to class index
                    all_labels = torch.argmax(all_labels, dim=1)
                    # Calculate accuracy
                    correct = (predicted == all_labels).sum().item()
                    epoch_accuracy = correct / all_labels.size(0)

            if validation_samples_per_class != 0:
                print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {epoch_accuracy:.4f}, Training Loss: {epoch_loss:.2f}, Validation Accuracy: {epoch_accuracy}, Validation Loss: {total_loss / total_val:.2f}")

            else:
                print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {epoch_accuracy:.4f}, Training Loss: {epoch_loss:.2f}")
                
            
        hist = TorchHistory()
        hist.history = {"accuracy": train_accuracy, "val_accuracy": val_accuracy, "loss": train_loss, "val_loss": val_loss}
        return hist

    def predict(self, X, batch_size=1, verbose=0):
        X = torch.tensor(X)
        # Move the input to the GPU
        X = X.to(self.device)
        self.model.eval()
        X_embeddings = self.get_embeddings(X)
        y_pred = self.model(X_embeddings)
        return y_pred.detach().numpy()
    
    def predict_raw(self, X, training=True):

        if training:
            self.model.train()
        else:
            self.model.eval()
        return self.predict(X)
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        return None

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        return None


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
