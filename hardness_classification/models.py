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
        model.add(Dense(512, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.5))
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


class OpticalFlowSVM:
    def __init__(self):
        self.model_name = "OpticalFlowSVM"
        self.model = None
        self.max_features = 500  # Define the maximum number of features

    def get_model(self, **kwargs):
        self.model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
        return self.model

    def compile_model(self, **kwargs):
        return None

    def compute_dense_optical_flow(self, video_frames):
        video_frames = video_frames.astype(np.uint8)
        flow_features = np.array([])
        for i in range(len(video_frames) - 1):
            prev_frame = cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(video_frames[i + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_features = np.concatenate([flow_features, flow.flatten()])
        return flow_features

    def detect_initial_points(self, image, display=False):
        """Detect the initial positions of the dots using blob detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0
        params.filterByArea = True
        params.minArea = 15
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return points

    def play_video(self, video_frames):
        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(video_frames[0].astype(np.uint8))

        def update(frame):
            im.set_array(video_frames[frame].astype(np.uint8))
            return (im,)

        ani = FuncAnimation(
            fig, update, frames=len(video_frames), interval=500, blit=True
        )
        plt.show()

    def compute_optical_flow(self, video_frames, display=False):
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        flow_features = np.array([])
        # Detect initial points of interest
        initial_frame = video_frames[0].astype(np.uint8)
        initial_frame_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)

        # Use blob detection to detect the initial points
        # p0 = self.detect_initial_points(video_frames[0].astype(np.uint8), display=display)
        # p0 = p0.reshape(-1, 1, 2)

        # Use goodFeaturesToTrack to detect the initial points
        p0 = cv2.goodFeaturesToTrack(initial_frame_gray, mask=None, **feature_params)

        # Display the initial points
        if display:
            points = p0.reshape(-1, 2)
            for point in points:
                cv2.circle(initial_frame, tuple(map(int, point)), 5, (0, 0, 255), -1)
            plt.imshow(initial_frame)
            plt.show()

        for i in range(len(video_frames) - 1):
            prev_frame = video_frames[i].astype(
                np.uint8
            )  # Ensure the frame is in the correct format
            next_frame = video_frames[i + 1].astype(
                np.uint8
            )  # Ensure the frame is in the correct format

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

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

            # Display the optical flow
            if display:
                for j, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)

                    # Calculate direction vector
                    direction = np.array([a - c, b - d])

                    # Scale the direction vector to increase line length by a factor of 5
                    scaled_direction = direction * 20

                    # Calculate new endpoint coordinates
                    c_new, d_new = c + scaled_direction[0], d + scaled_direction[1]

                    # Draw the line with increased length
                    cv2.line(
                        video_frames[i],
                        (a, b),
                        (int(c_new), int(d_new)),
                        (0, 0, 255),
                        2,
                    )
                    # Draw the initial points and the new points
                    cv2.circle(video_frames[i], (c, d), 1, (255, 0, 0), -1)
                    cv2.circle(video_frames[i], (a, b), 1, (0, 255, 0), -1)
                self.play_video(video_frames)

            # Update the initial points
            p0 = good_new.reshape(-1, 1, 2)

        # Pad or truncate to max_features length
        if flow_features.shape[0] < self.max_features:
            flow_features = np.pad(
                flow_features, (0, self.max_features - len(flow_features)), "constant"
            )
        else:
            flow_features = flow_features[: self.max_features]

        return flow_features

    def train_model(
        self,
        X_train,
        y_train,
        validation_samples_per_class,
        train_samples_per_class,
        **kwargs,
    ):
        X_features_original = [self.compute_optical_flow(video) for video in X_train]
        X_features = np.array(X_features_original)
        y_train = np.argmax(y_train, axis=1)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_features,
            y_train,
            train_size=train_samples_per_class
            / (train_samples_per_class + validation_samples_per_class),
            stratify=y_train,
        )

        self.model.fit(X_train, y_train)

        # Evaluate the classifier
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)

        return [train_accuracy, val_accuracy]

    def predict(self, X):
        X_features = [self.compute_optical_flow(video) for video in X]
        return self.model.predict(X_features)

    def save_model(self, path):
        joblib.dump(self.model, path)
        return True


class OpticalFlowRegression:
    def __init__(self):
        self.model_name = "OpticalFlowRegression"
        self.model = None
        self.max_features = 500  # Define the maximum number of features

    def get_model(self, **kwargs):
        pass

    def compile_model(self, **kwargs):
        pass

    def compute_optical_flow(self, video_frames):
        return compute_dense_optical_flow(video_frames, max_features=self.max_features)

    def compute_dense_optical_flow(self, video_frames):
        video_frames = video_frames.astype(np.uint8)
        flow_features = np.array([])
        for i in range(len(video_frames) - 1):
            prev_frame = cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(video_frames[i + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_features = np.concatenate([flow_features, flow.flatten()])
        return flow_features

    def train_model(
        self,
        X_train,
        y_train,
        validation_samples_per_class,
        train_samples_per_class,
        **kwargs,
    ):
        # Compute the optical flow features (concatenated) for each video in the training set
        X_features_original = [self.compute_optical_flow(video) for video in X_train]
        X_features = np.array(X_features_original)
        y_train = np.argmax(y_train, axis=1)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_features,
            y_train,
            train_size=train_samples_per_class
            / (train_samples_per_class + validation_samples_per_class),
            stratify=y_train,
        )

        # Compute the average optical flow features for each class
        X_train_sum = np.sum(X_train, axis=1)
        y_train_onehot = utils.to_categorical(y_train)
        X_train_avg_class = (y_train_onehot.T @ X_train_sum) / np.sum(
            y_train_onehot, axis=0
        )
        self.model = X_train_avg_class

        # Evaluate the classifier
        y_pred_train = [
            np.argmin(abs(X_train_avg_class - X_test)) for X_test in X_train_sum
        ]
        y_pred_val = [
            np.argmin(abs(X_train_avg_class - X_test))
            for X_test in np.sum(X_val, axis=1)
        ]
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        return [train_accuracy, val_accuracy]

    def predict(self, X):
        X_features = [self.compute_optical_flow(video) for video in X]
        X_features_sum = np.sum(X_features, axis=1)
        # The prediction is the class closest to the sum of the optical flow features from the X_features_avg_class
        predictions = [np.argmin(abs(self.model - X_test)) for X_test in X_features_sum]
        return np.array(predictions)


class ConvLSTMSiamese:
    def __init__(self):
        self.model_name = "ConvLSTMSiamese"
        self.model = None
        self.base_model = None
        self.X_classes = None
        self.class_labels = None
        self.max_triplets = 100

    def get_base_model(self, input_shape):
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

        # Fourth Layer
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

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))

        self.base_model = model
        return model

    def get_model(self, input_shape, num_classes):
        input_anchor = Input(shape=input_shape, name="anchor_input")
        input_positive = Input(shape=input_shape, name="positive_input")
        input_negative = Input(shape=input_shape, name="negative_input")

        base_network = self.get_base_model(input_shape)

        embed_anchor = base_network(input_anchor)
        embed_positive = base_network(input_positive)
        embed_negative = base_network(input_negative)

        self.model = Model(
            inputs=[input_anchor, input_positive, input_negative],
            outputs=[embed_anchor, embed_positive, embed_negative],
        )

        return self.model

    def triplet_loss(self, y_pred, alpha=0.2):
        # import pdb; pdb.set_trace()
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        pos_dist = K.sum(K.square(anchor - positive))
        neg_dist = K.sum(K.square(anchor - negative))

        basic_loss = pos_dist - neg_dist + alpha
        loss = K.maximum(basic_loss, 0.0)
        return loss

    def compile_model(self, optimizer, **kwargs):
        # optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss=self.triplet_loss)

    def create_triplets(self, X_train, y_train):
        # Dictionary to store class-wise samples
        class_samples = {}
        for class_label in np.unique(y_train):
            class_samples[class_label] = X_train[y_train == class_label]

        X_anchor = []
        X_positive = []
        X_negative = []

        # for class_label in np.unique(y_train):
        #     positive_samples = class_samples[class_label]
        #     negative_samples = np.concatenate([class_samples[other_label]
        #                                       for other_label in class_samples if other_label != class_label])

        #     for i in range(len(positive_samples)):
        #         anchor = positive_samples[i]
        #         for pos_idx in [j for j in range(len(positive_samples)) if j != i]:
        #            positive = positive_samples[pos_idx]
        #            for neg_idx in range(len(negative_samples)):
        #               negative = negative_samples[neg_idx]
        #               X_anchor.append(anchor)
        #               X_positive.append(positive)
        #               X_negative.append(negative)

        # X_anchor = np.array(X_anchor)
        # X_positive = np.array(X_positive)
        # X_negative = np.array(X_negative)

        # # Select n_max triplets
        # n_max = min(self.max_triplets, X_anchor.shape[0])
        # idx = np.random.choice(X_anchor.shape[0], n_max, replace=False)
        # X_anchor = X_anchor[idx]
        # X_positive = X_positive[idx]
        # X_negative = X_negative[idx]

        # Create n_max triplets
        for i in range(self.max_triplets):
            class_label = np.random.choice(y_train)
            anchor = class_samples[class_label][
                np.random.choice(len(class_samples[class_label]))
            ]
            positive = class_samples[class_label][
                np.random.choice(len(class_samples[class_label]))
            ]
            negative_class = np.random.choice(
                [
                    other_class
                    for other_class in class_samples
                    if other_class != class_label
                ]
            )
            negative = class_samples[negative_class][
                np.random.choice(len(class_samples[negative_class]))
            ]
            X_anchor.append(anchor)
            X_positive.append(positive)
            X_negative.append(negative)

        X_anchor = np.array(X_anchor)
        X_positive = np.array(X_positive)
        X_negative = np.array(X_negative)

        return X_anchor, X_positive, X_negative

    def store_class_samples(self, X_train, y_train):
        self.X_classes = [
            X_train[y_train == class_label] for class_label in np.unique(y_train)
        ]
        self.class_labels = np.unique(y_train)

    def train_model(
        self,
        X_train,
        y_train,
        batch_size,
        epochs,
        validation_samples_per_class,
        train_samples_per_class,
    ):
        y_train = np.argmax(y_train, axis=1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            train_size=train_samples_per_class
            / (train_samples_per_class + validation_samples_per_class),
            stratify=y_train,
        )
        self.store_class_samples(X_train, y_train)
        X_anchor, X_positive, X_negative = self.create_triplets(X_train, y_train)
        y_dummy = np.zeros((X_anchor.shape[0], 3))
        history = self.model.fit(
            [X_anchor, X_positive, X_negative],
            y_dummy,
            batch_size=batch_size,
            epochs=epochs,
        )
        # Evaluate the classifier on each sample in the training set
        train_predictions = []
        for sample in X_train:
            sample = np.expand_dims(sample, axis=0)
            train_predictions.append(self.predict(sample))

        train_predictions = np.array(train_predictions)
        training_accuracy = accuracy_score(y_train, train_predictions)

        # Evaluate the classifier on each class in the validation set
        val_predictions = []
        for sample in X_val:
            sample = np.expand_dims(sample, axis=0)
            val_predictions.append(self.predict(sample))
        val_predictions = np.array(val_predictions)
        validation_accuracy = accuracy_score(y_val, val_predictions)

        train_loss = history.history["loss"][-1]
        val_loss = (
            history.history["val_loss"][-1] if "val_loss" in history.history else np.nan
        )
        return [training_accuracy, validation_accuracy]

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(
            path, custom_objects={"triplet_loss": self.triplet_loss}
        )
        self.base_model = Model(
            inputs=self.model.input[0],
            outputs=self.model.get_layer("sequential").output,
        )
        return None

    def get_embedding(self, X):
        return self.base_model.predict(X)

    def predict(self, X_test):
        test_embedding = self.get_embedding(X_test)
        class_embeddings = [self.get_embedding(X_class) for X_class in self.X_classes]

        distances = [
            euclidean_distances(test_embedding, class_embedding)
            for class_embedding in class_embeddings
        ]
        avg_distances = [np.mean(distance) for distance in distances]

        closest_class_idx = np.argmin(avg_distances)
        return self.class_labels[closest_class_idx]


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
        linear = torch.nn.Linear(
            self.backbone.num_features * input_shape[0], num_classes
        )
        # Add dropout layer
        dropout = torch.nn.Dropout(0.2)
        self.model = torch.nn.Sequential(dropout, linear)
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
            validation_split = None
        else:
            validation_split = validation_samples_per_class / (
                validation_samples_per_class + train_samples_per_class
            )

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_split,
            stratify=y_train,
        )
        print("Train samples: ", X_train.shape[0], "Validation samples: ", X_val.shape[0])
        # Convert the data to PyTorch tensors and move to GPU
        X_train = torch.tensor(X_train)
        X_train = X_train.to(self.device)
        X_train_embeddings = self.get_embeddings(X_train)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_train = y_train.to(self.device)
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
        val_dataset = torch.utils.data.TensorDataset(X_val_embeddings, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        # Train the model
        for epoch in range(epochs):
            self.model.train()
            total_train = 0
            correct_train = 0
            total_loss = 0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.loss(y_pred, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                _, predicted = torch.max(y_pred, 1)
                total_train += X_batch.size(0)
                correct_train += (predicted == y_batch).sum().item()
                
            train_accuracy.append(correct_train / total_train)
            train_loss.append(total_loss / total_train)
            print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {correct_train / total_train:.4f}, Training Loss: {total_loss / total_train:.4f}")
            # Evaluate the model on the validation set
            self.model.eval()

            if validation_samples_per_class != 0:
                total_val = 0
                correct_val = 0
                val_loss = 0
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    _, predicted = torch.max(y_pred, 1)
                    total_val += X_batch.size(0)
                    correct_val += (predicted == y_batch).sum().item()
                    val_loss += self.loss(y_pred, y_batch).item()
                
                val_accuracy.append(correct_val / total_val)
                val_loss.append(val_loss / total_val)
                print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {correct_val / total_val:.4f}, Validation Loss: {val_loss / total_val:.4f}")
                
            
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


class DINOv2SVM:
    def __init__(self, dino_version="dinov2_vits14"):
        self.model_name = "DINOv2SVM"
        self.model = None
        self.device = "cpu"
        self.dino_version = dino_version

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
        self.backbone = torch.hub.load("facebookresearch/dinov2", self.dino_version)
        self.patch_size = self.backbone.patch_size
        self.backbone.eval()
        self.model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
        return self.model

    def pad_to_multiple_of_patch_size(self, X, patch_size):
        _, height, width, _ = X.shape
        pad_height = (patch_size - height % patch_size) % patch_size
        pad_width = (patch_size - width % patch_size) % patch_size
        return torch.nn.functional.pad(X, (0, 0, 0, pad_width, 0, pad_height))

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
        pass

    def train_model(
        self,
        X_train,
        y_train,
        batch_size,
        epochs,
        validation_samples_per_class,
        train_samples_per_class,
    ):
        # Move the backbone to the GPU
        self.backbone = self.backbone.to(self.device)
        y_train = np.argmax(y_train, axis=1)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            train_size=train_samples_per_class
            / (train_samples_per_class + validation_samples_per_class),
            stratify=y_train,
        )
        # Convert the data to PyTorch tensors and move to GPU
        X_train = torch.tensor(X_train)
        X_train = X_train.to(self.device)
        X_train_embeddings = self.get_embeddings(X_train)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_train = y_train.to(self.device)
        X_val = torch.tensor(X_val)
        X_val = X_val.to(self.device)
        X_val_embeddings = self.get_embeddings(X_val)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        y_val = y_val.to(self.device)

        # Train the model
        self.model.fit(X_train_embeddings, y_train)

        # Evaluate the classifier
        y_pred_train = self.model.predict(X_train_embeddings)
        y_pred_val = self.model.predict(X_val_embeddings)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        return [train_accuracy, val_accuracy]

    def predict(self, X):
        X = torch.tensor(X)
        # Move the input to the GPU
        X = X.to(self.device)
        X_embeddings = self.get_embeddings(X)
        y_pred = self.model.predict(X_embeddings.cpu().detach().numpy())
        return y_pred.astype(int)

    def save_model(self, path):
        joblib.dump(self.model, path)
        return True

    def load_model(self, path):
        self.model = joblib.load(path)
        return None


class OpticalFlowRatio:
    def __init__(self):
        self.model_name = "OpticalFlowRatio"
        self.model = None
        self.max_features = 500

    def get_model(self, **kwargs):
        pass

    def compile_model(self, **kwargs):
        pass

    def detect_initial_points(self, image, display=False):
        """Detect the initial positions of the dots using blob detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0
        params.filterByArea = True
        params.minArea = 15
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # Display the detected points
        if display:
            for point in points:
                cv2.circle(image, tuple(map(int, point)), 5, (0, 0, 255), -1)
            plt.imshow(image)
            plt.show()
        return points

    def play_video(self, video_frames):
        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(video_frames[0].astype(np.uint8))

        def update(frame):
            im.set_array(video_frames[frame].astype(np.uint8))
            return (im,)

        ani = FuncAnimation(
            fig, update, frames=len(video_frames), interval=500, blit=True
        )
        plt.show()

    def estimate_K(self, video_frames, display=False):
        displacements = []

        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Detect initial points
        p0 = self.detect_initial_points(video_frames[0].astype(np.uint8))

        # Convert the first image to grayscale
        prev_gray = cv2.cvtColor(video_frames[0].astype(np.uint8), cv2.COLOR_BGR2GRAY)

        for i in range(1, len(video_frames)):
            displacements_per_frame = []
            next_gray = cv2.cvtColor(
                video_frames[i].astype(np.uint8), cv2.COLOR_BGR2GRAY
            )

            # Calculate optical flow using Lucas-Kanade method
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, p0, None, **lk_params
            )

            # Select good points
            st = st.reshape(-1)
            good_initial_points = p0[st == 1]
            good_new_points = p1[st == 1]

            # Compute displacement
            for p0, p1 in zip(good_initial_points, good_new_points):
                displacement = np.linalg.norm(p1 - p0)
                displacements_per_frame.append(displacement)
            displacements.append(np.mean(displacements_per_frame))

            # Display the optical flow
            if display:
                for j, (new, old) in enumerate(
                    zip(good_new_points, good_initial_points)
                ):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)

                    # Calculate direction vector
                    direction = np.array([a - c, b - d])

                    # Scale the direction vector to increase line length by a factor of 5
                    scaled_direction = direction * 20

                    # Calculate new endpoint coordinates
                    c_new, d_new = c + scaled_direction[0], d + scaled_direction[1]

                    # Draw the line with increased length
                    cv2.line(
                        video_frames[i],
                        (a, b),
                        (int(c_new), int(d_new)),
                        (0, 0, 255),
                        2,
                    )
                    # Draw the initial points and the new points
                    cv2.circle(video_frames[i], (c, d), 1, (255, 0, 0), -1)
                    cv2.circle(video_frames[i], (a, b), 1, (0, 255, 0), -1)
                self.play_video(video_frames)

            # Update the initial points
            p0 = good_new_points.reshape(-1, 1, 2)
            prev_gray = next_gray
        total_displacement = np.sum(displacements)
        diff_displacement = np.diff(displacements)
        estimate_K = total_displacement
        return estimate_K

    def train_model(
        self,
        X_train,
        y_train,
        validation_samples_per_class,
        train_samples_per_class,
        **kwargs,
    ):
        X_features = [self.estimate_K(video) for video in X_train]
        X_features = np.array(X_features)
        y_train = np.argmax(y_train, axis=1)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_features,
            y_train,
            train_size=train_samples_per_class
            / (train_samples_per_class + validation_samples_per_class),
            stratify=y_train,
        )

        # Compute the average K for each class
        y_train_onehot = utils.to_categorical(y_train)
        X_train_avg_class = (y_train_onehot.T @ X_train) / np.sum(
            y_train_onehot, axis=0
        )
        self.model = X_train_avg_class

        # Evaluate the classifier
        y_pred_train = [np.argmin(abs(self.model - X_test)) for X_test in X_train]
        y_pred_val = [np.argmin(abs(self.model - X_test)) for X_test in X_val]
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)

        return self.model

    def predict(self, X):
        X_features = [self.estimate_K(video) for video in X]
        return np.array([np.argmin(abs(self.model - X_test)) for X_test in X_features])


class DINOv2MLP:
    def __init__(self, dino_version="dinov2_vits14"):
        self.model_name = "DINOv2MLP"
        self.model = None
        self.device = "cpu"
        self.dino_version = dino_version

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

    def get_model(self, input_shape):
        self.backbone = torch.hub.load("facebookresearch/dinov2", self.dino_version)
        self.patch_size = self.backbone.patch_size
        self.backbone.eval()
        self.model = torch.nn.Linear(self.backbone.num_features * input_shape[0], 1)
        return self.model

    def pad_to_multiple_of_patch_size(self, X, patch_size):
        _, height, width, _ = X.shape
        pad_height = (patch_size - height % patch_size) % patch_size
        pad_width = (patch_size - width % patch_size) % patch_size
        return torch.nn.functional.pad(X, (0, 0, 0, pad_width, 0, pad_height))

    def get_embeddings(self, X):
        with torch.no_grad():
            n_samples, n_frames, height, width, channels = X.shape
            X_reshaped = X.reshape(-1, height, width, channels)
            X_padded = self.pad_to_multiple_of_patch_size(
                X_reshaped, patch_size=self.patch_size
            )
            X_normalized = X_padded / 255.0
            X_permuted = X_normalized.permute(0, 3, 1, 2)
            embeddings = self.backbone(X_permuted.float())
            embedding_dim = embeddings.shape[1]
            embeddings = embeddings.reshape(n_samples, n_frames * embedding_dim)
            return embeddings

            # for i in range(X.shape[1]):
            #     # self.visualize_dino_features(X[:,i,:,:,:])
            #     X_resize = self.pad_to_multiple_of_patch_size(X[:,:,i,:,:,:], patch_size=self.patch_size)
            #     X_normalize = X_resize / 255.0
            #     X_permute = X_normalize.permute(0, 1, 4, 2, 3)
            #     embedding = self.backbone(X_permute.float())
            #     embeddings = torch.cat((embeddings, embedding), dim=1)
            # if i == 0:
            #     embeddings = embedding
            # else:
            #     embeddings = embeddings + embedding
        # return embeddings

    def triplet_loss(self, y_pred, y_triplets, margin=1.0):
        # For each triplet in X_triplet, the third element is the anchor, if y_triplet is 0 then the first element is the positive sample, otherwise the second element is the positive sample
        anchor = y_pred[:, 2]
        positive = y_pred[:, 0] * y_triplets + y_pred[:, 1] * (1 - y_triplets)
        negative = y_pred[:, 1] * y_triplets + y_pred[:, 0] * (1 - y_triplets)
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        triplet_loss = torch.nn.functional.relu(dist_pos - dist_neg + margin)
        return torch.mean(triplet_loss)

    def pair_loss(self, y_pred, y_pairs, margin):
        # For each prediction, if y_pairs is 0 then the first element should be higher than the second element, otherwise the second element should be higher than the first element
        higher = y_pred * y_pairs + (1 - y_pred) * (1 - y_pairs)
        lower = y_pred * (1 - y_pairs) + (1 - y_pred) * y_pairs
        margin_loss = torch.nn.functional.relu(margin - (higher - lower))
        return margin_loss.mean()

    def compile_model(self, loss, metrics, **kwargs):
        alpha = 0.5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.metrics = metrics
        return None

    def train_model(
        self,
        X_pairs_train,
        y_pairs_train,
        X_pairs_val,
        y_pairs_val,
        X_triplets_train,
        y_triplets_train,
        X_triplets_val,
        y_triplets_val,
        batch_size,
        epochs,
        validation_samples_per_class,
        train_samples_per_class,
    ):
        # Move the model to the GPU
        self.model = self.model.to(self.device)
        # Move the backbone to the GPU
        self.backbone = self.backbone.to(self.device)

        # Convert the data to PyTorch tensors and move to GPU
        X_pairs_train = torch.tensor(X_pairs_train)
        X_pairs_train = X_pairs_train.to(self.device)
        n_pairs, n_samples, n_frames, height, width, channels = X_pairs_train.shape
        X_pairs_train_embeddings = self.get_embeddings(
            X_pairs_train.reshape(-1, n_frames, height, width, channels)
        ).reshape(n_pairs, n_samples, -1)
        y_pairs_train = torch.tensor(y_pairs_train, dtype=torch.float32)
        y_pairs_train = y_pairs_train.to(self.device)
        X_pairs_val = torch.tensor(X_pairs_val)
        X_pairs_val = X_pairs_val.to(self.device)
        n_pairs, n_samples, n_frames, height, width, channels = X_pairs_val.shape
        X_pairs_val_embeddings = self.get_embeddings(
            X_pairs_val.reshape(-1, n_frames, height, width, channels)
        ).reshape(n_pairs, n_samples, -1)
        y_pairs_val = torch.tensor(y_pairs_val, dtype=torch.float32)
        y_pairs_val = y_pairs_val.to(self.device)
        X_triplets_train = torch.tensor(X_triplets_train)
        X_triplets_train = X_triplets_train.to(self.device)
        n_triplets, n_samples, n_frames, height, width, channels = (
            X_triplets_train.shape
        )
        X_triplets_train_embeddings = self.get_embeddings(
            X_triplets_train.reshape(-1, n_frames, height, width, channels)
        ).reshape(n_triplets, n_samples, -1)
        y_triplets_train = torch.tensor(y_triplets_train, dtype=torch.float32)
        y_triplets_train = y_triplets_train.to(self.device)
        X_triplets_val = torch.tensor(X_triplets_val)
        n_triplets, n_samples, n_frames, height, width, channels = X_triplets_val.shape
        X_triplets_val = X_triplets_val.to(self.device)
        X_triplets_val_embeddings = self.get_embeddings(
            X_triplets_val.reshape(-1, n_frames, height, width, channels)
        ).reshape(n_triplets, n_samples, -1)
        y_triplets_val = torch.tensor(y_triplets_val, dtype=torch.float32)
        y_triplets_val = y_triplets_val.to(self.device)

        # Create a DataLoader for the training and validation sets
        pairs_train_dataset = torch.utils.data.TensorDataset(
            X_pairs_train_embeddings, y_pairs_train
        )
        pairs_train_loader = torch.utils.data.DataLoader(
            pairs_train_dataset, batch_size=batch_size, shuffle=True
        )
        pairs_val_dataset = torch.utils.data.TensorDataset(
            X_pairs_val_embeddings, y_pairs_val
        )
        pairs_val_loader = torch.utils.data.DataLoader(
            pairs_val_dataset, batch_size=batch_size, shuffle=False
        )
        triplets_train_dataset = torch.utils.data.TensorDataset(
            X_triplets_train_embeddings, y_triplets_train
        )
        triplets_train_loader = torch.utils.data.DataLoader(
            triplets_train_dataset, batch_size=batch_size, shuffle=True
        )
        triplets_val_dataset = torch.utils.data.TensorDataset(
            X_triplets_val_embeddings, y_triplets_val
        )
        triplets_val_loader = torch.utils.data.DataLoader(
            triplets_val_dataset, batch_size=batch_size, shuffle=False
        )

        pair_loss_weight = 0.5
        triplet_loss_weight = 0.5
        same_class_weight = 0.5

        # Train the model
        for epoch in range(epochs):
            self.model.train()
            for X_batch, y_batch in pairs_train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                pair_loss = pair_loss_weight * self.pair_loss(
                    y_pred, y_batch, margin=1.0
                )

            for X_batch_triplet, y_batch_triplet in triplets_train_loader:
                y_pred_triplet = self.model(X_batch_triplet)
                triplet_loss = triplet_loss_weight * self.triplet_loss(
                    y_pred_triplet, y_batch_triplet, margin=1.0
                )

            loss = pair_loss_weight * pair_loss + triplet_loss_weight * triplet_loss
            loss.backward()
            self.optimizer.step()

            # Evaluate the model on the validation set
            self.model.eval()
            pair_val_predictions = []
            for X_batch, y_batch in pairs_val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                pair_val_predictions.extend(torch.argmax(y_pred, dim=1).tolist())
            pair_val_accuracy = accuracy_score(y_pairs_val, pair_val_predictions)
            print(
                f"Epoch {epoch+1}/{epochs}, Pair Validation Accuracy: {pair_val_accuracy:.4f}"
            )
            triplet_val_predictions = []
            for X_batch, y_batch in triplets_val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                triplet_val_predictions.extend(torch.argmax(y_pred, dim=1).tolist())
            triplet_val_accuracy = accuracy_score(
                y_triplets_val, triplet_val_predictions
            )
            print(
                f"Epoch {epoch+1}/{epochs}, Triplet Validation Accuracy: {triplet_val_accuracy:.4f}"
            )
            val_accuracy = (pair_val_accuracy + triplet_val_accuracy) / 2
            print(
                f"Epoch {epoch+1}/{epochs}, Average Validation Accuracy: {val_accuracy:.4f}"
            )

        # Compute the training accuracy
        pair_train_predictions = []
        y_pred = self.model(X_pairs_train_embeddings)
        pair_train_predictions.extend(torch.argmax(y_pred, dim=1).tolist())
        pair_train_accuracy = accuracy_score(y_pairs_train, pair_train_predictions)
        triplet_train_predictions = []
        y_pred = self.model(X_triplets_train_embeddings)
        triplet_train_predictions.extend(torch.argmax(y_pred, dim=1).tolist())
        triplet_train_accuracy = accuracy_score(
            y_triplets_train, triplet_train_predictions
        )
        train_accuracy = (pair_train_accuracy + triplet_train_accuracy) / 2
        return [train_accuracy, val_accuracy]

    def predict(self, X):
        X = torch.tensor(X)
        # Move the input to the GPU
        X = X.to(self.device)
        self.model.eval()
        X_embeddings = self.get_embeddings(X)
        y_pred = self.model(X_embeddings)
        return y_pred.detach().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        return None

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        return None


# ================================================= Spatio Temporal Convolution ======================================================
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/video/video_classification.ipynb


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
