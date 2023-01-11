from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from os.path import join
import json

from sklearn.model_selection import train_test_split

IMG_SIZE_0 = 240
IMG_SIZE_1 = 320
IMG_CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 200
NUM_FEATURES = 2048
SEQ_LENGTH = 5


def select_frames(frames,k):

    frame_num = frames.shape[0]

    neutral = frames[int(frame_num * .05)] # neural deformation before pressing

    dis = lambda x: np.linalg.norm( x- neutral)
    distances = np.array(list(map(dis,frames)))


    epsilon = 1400

    first_deformation_index = np.argmax(distances > epsilon)
    max_deformation_index = np.argmax(distances)

    time_span = max_deformation_index - first_deformation_index
    indizes = [int(np.round(first_deformation_index + i *time_span/(k-1))) for i in range(k)]
  

    return frames[indizes]



"""
This function extracts the essary data from given data folder.
@param path/to/data
@return (X,Y) with
    X.shape == (n,5,128,128)
    len(Y) = n
"""


def create_dataset(path):
    X = []
    Y = []
    meta_path = join(path, "meta/")
    frames_path = join(path, "frames/")

    for file in os.listdir(meta_path):
        meta = None
        with open(join(meta_path, file),"r") as f:
            meta = json.load(f)

        Y.append(meta["object"])

        frames = np.load(join(frames_path, meta["id"] + "_frames.npy"))
        X.append(select_frames(frames, SEQ_LENGTH))
       

    X = np.array(X)
    y = label_processor(Y).numpy()
    print(X.shape)
 

    return X,y   

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE_0,IMG_SIZE_1, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE_0, IMG_SIZE_1, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(["hardness" + str(i +1) for i in range(5)])
)




def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((SEQ_LENGTH, NUM_FEATURES))


    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model





def run_experiment():
    #create dataset
    """
    X, y = create_dataset("data/")

    with open("x.npy", "wb") as f:
        np.save(f,X)

    with open("y.npy", "wb") as f:
        np.save(f,y)
    """

    X = np.load("X.npy")
    y = np.load("y.npy")

    print(X[0,0].shape)
    

    # preprocess
    temp = np.zeros((X.shape[0], SEQ_LENGTH, NUM_FEATURES))
    for i in range(X.shape[0]):
        for j in range(SEQ_LENGTH):
            temp[i,j,:] = feature_extractor.predict(X[None,i,j])

    X = temp
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 42)

    # train



    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()


if __name__ == "__main__":
   run_experiment()