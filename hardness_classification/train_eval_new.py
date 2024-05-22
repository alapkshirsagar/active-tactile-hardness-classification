from preprocessing import *
from models import *

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.math import confusion_matrix
from tensorflow.keras.models import load_model
import numpy as np
import itertools
from time import time
import datetime
from log import Logger
import uuid
import pandas as pd
from utils import *

DATASET_ID = "42"
CLASSES = ["hardness" + str(i+1) for i in range(5)]
MODEL_DUMP_PATH = "model_dumps/"
LOG_PATH = "logs/"



def run_initial_sample_per_class(model_creator, manager, save_path, num_additional_epochs=20):
    confusion = np.zeros((len(CLASSES), len(CLASSES)))
    lossfunction = "categorical_crossentropy"
    
    X_base, y_base = manager.get_raw_with_classes(CLASSES, 1)
    X_test, y_test = manager.get_raw_with_classes(CLASSES, 1)

    input_shape = X_base.shape[1:]
    num_classes = len(CLASSES)
    model = model_creator(input_shape, num_classes)
    model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])
    model.fit(
            X_base, y_base, epochs=1, batch_size=1,
            validation_split=0,
        )
    loss, acc = model.evaluate(X_test, y_test, batch_size=1)
    print("test_loss", loss)
    print("test_acc", acc)
    # Continue training with samples from random classes for num_additional_epochs
    for _ in range(num_additional_epochs):
        random_class = random.choice(CLASSES)
        X_additional, y_additional = manager.get_raw_with_classes([random_class], 1)
        model.fit(X_additional, y_additional, epochs=1, batch_size=1, verbose=0)
        X_test, y_test = manager.get_raw_with_classes(CLASSES, 1)
        loss, acc = model.evaluate(X_test, y_test, batch_size=1)
        print("test_loss", loss)
        print("test_acc", acc)

    model.summary()


    # Predict on all classes
    for class_index, class_name in enumerate(CLASSES):
        X_comp, _ = manager.get_raw_with_classes([class_name], 10)

        dist = model.predict(X_comp)
        choice = np.argmax(dist, axis=1)
        prediction = np.zeros((choice.size, len(CLASSES)))
        prediction[np.arange(choice.size), choice] = 1

        row = prediction.sum(axis=0)
        confusion[class_index, :] = row

    print(confusion)

if __name__ == "__main__":
    manager = DatasetManager(DATASET_ID, ["C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"], CLASSES)
    run_initial_sample_per_class(model_creator=get_ConvLSTM, 
                                                manager=manager,
                                                save_path="model_dumps/newstrats/")