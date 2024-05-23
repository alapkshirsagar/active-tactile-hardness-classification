import numpy as np
import random
import uuid
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.math import confusion_matrix
import itertools
from time import time
import datetime
from log import Logger
import pandas as pd
from utils import *
from models import get_ConvLSTM
from preprocessing import DatasetManager
from os.path import join  # Added import for join

DATASET_ID = "42"
CLASSES = ["hardness" + str(i + 1) for i in range(5)]
MODEL_DUMP_PATH = "model_dumps/"
LOG_PATH = "logs/"

def calculate_entropy_with_dropout(model, X_val, num_samples=10):
    preds = np.array([model(X_val, training=True) for _ in range(num_samples)])
    mean_preds = np.mean(preds, axis=0)
    entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-9), axis=1)
    return np.mean(entropy)


def run_with_strat(model_creator, manager, save_path, n_experiments=1, epochs=200,validation_size = 4,  strat="random"):
    overall_confusion = np.zeros((len(CLASSES), len(CLASSES)))

    lossfunction = "categorical_crossentropy"
    logger = Logger(join(LOG_PATH, "leave_one_out.txt"),
                    time(), DATASET_ID, "full_dataset",
                    model_creator.__name__,
                    "for save path see below",
                    lossfunction=lossfunction)
    
    # Create validation set, validation_size samples per class
    validation_set = {} # Dictionary to store validation set
    if strat == "entropy" or strat == "variance":
            for cls in CLASSES: # Calculate entropy for each class
                X_val_cls, y_val_cls = manager.get_raw_with_classes([cls], validation_size) # Get one sample from each class
                y_val_cls_full = np.zeros((y_val_cls.shape[0], len(CLASSES))) # Adjust y_val_cls to reflect all classes
                class_idx = CLASSES.index(cls) # Get the index of the class
                y_val_cls_full[:, class_idx] = y_val_cls[:, 0] # Set the index of the class to the value of y_val_cls
                y_val_cls = y_val_cls_full # Set y_val_cls to the full value
                validation_set[cls] = (X_val_cls, y_val_cls)

    for test_class_index, test_class in enumerate(CLASSES):
        print("Test class: ", test_class)
        training_classes = [c for c in CLASSES if c != test_class]
        print("Training classes: ", training_classes)
        X_train, y_train = manager.get_raw_with_classes(training_classes, 1, seed=42)

        # Ensure y_train covers all classes and adjust labels
        y_train_full = np.zeros((y_train.shape[0], len(CLASSES)))
        for i, c in enumerate(training_classes):
            class_idx = CLASSES.index(c)
            y_train_full[:, class_idx] = y_train[:, i]
        y_train = y_train_full

        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]

        model = model_creator(input_shape, num_classes)
        model.summary()

        model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])

        # initial training with 1 sample per class
        model.fit(X_train, y_train, epochs=1, batch_size=4, callbacks=[])

        

        # further training with random samples, one example per epoch

        for epoch in range(epochs):
            selected_class = None 
            if strat == "random":
                selected_class = random.choice(training_classes)
            elif strat == "entropy": 
                entropies = []

                for cls in training_classes: # Calculate entropy for each class
                    X_val_cls, y_val_cls = validation_set[cls] # Get the validation set for the class
                    entropy = calculate_entropy_with_dropout(model, X_val_cls, num_samples=validation_size) # Calculate entropy
                    entropies.append(entropy)
                print("Entropies: ", entropies)
                selected_class = training_classes[np.argmax(entropies)]

             
            print("Epoch ", epoch + 1, "/", epochs, ", selected class: ", selected_class)
            X_train_epoch, y_train_epoch = manager.get_raw_with_classes([selected_class], 1)

            # Adjust y_train_epoch to reflect all classes
            y_train_epoch_full = np.zeros((y_train_epoch.shape[0], len(CLASSES)))
            class_idx = CLASSES.index(selected_class)
            y_train_epoch_full[:, class_idx] = y_train_epoch[:, 0]
            y_train_epoch = y_train_epoch_full

            model.fit(X_train_epoch, y_train_epoch, epochs=1, batch_size=4, callbacks=[])

        model_id = str(uuid.uuid4())
        try:
            model_save_path = f"{save_path}_{model_id}.h5"
            model.save(model_save_path)
        except:
            print("could not write")

        X_test, y_test = manager.get_raw_with_classes([test_class], 10, seed=42)

        # Adjust y_test to reflect all classes
        y_test_full = np.zeros((y_test.shape[0], len(CLASSES)))
        class_idx = CLASSES.index(test_class)
        y_test_full[:, class_idx] = y_test[:, 0]
        y_test = y_test_full

        dist = model.predict(X_test)
        choice = np.argmax(dist, axis=1)
        prediction = np.zeros((choice.size, len(CLASSES)))
        prediction[np.arange(choice.size), choice] = 1

        row = prediction.sum(axis=0)
        overall_confusion[test_class_index, :] += row

    print(overall_confusion)


if __name__ == "__main__":
    manager = DatasetManager(DATASET_ID, ["C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"], CLASSES)
    run_with_strat(get_ConvLSTM, manager, "model_dumps/newstrat", n_experiments=1, epochs=10, strat="entropy")
