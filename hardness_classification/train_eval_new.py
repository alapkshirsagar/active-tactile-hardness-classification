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

def run_with_strat(model_creator, manager, save_path, n_experiments = 1, epochs = 200, strat = "random"):
    confusion = np.zeros((len(CLASSES), len(CLASSES)))

    lossfunction = "categorical_crossentropy"
    logger = Logger(join(LOG_PATH, "leave_on_out.txt"),
        time(),DATASET_ID,"full_dataset",
        model_creator.__name__,
        "for save path see below",
        lossfunction=lossfunction)


    for i in range(n_experiments):

        X_base, y_base = manager.get_raw_with_classes([x for j,x in enumerate(CLASSES)], 1, seed = 42)
        y_base = y_base[:,[j for j in range(len(CLASSES)) ]]
        X_comp, _ = manager.get_raw_with_classes(CLASSES, 40, seed = 42)

        input_shape = X_base.shape[1:]
        num_classes = y_base.shape[1]

        model = model_creator(input_shape, num_classes)
        model.summary()
        X_train = X_base
        y_train = y_base
        # create test set from 40 samples each class
        X_test, y_test = manager.get_raw_with_classes(CLASSES, 40, seed = 64) # TODO: It's possible to have some intersection with the training set, is it acceptable?
        y_test = y_test[:,[j for j in range(len(CLASSES))]]

        stop_callback = EarlyStopping(monitor = "val_accuracy", patience = 30, mode = "max", restore_best_weights = True)

        model.compile(loss = lossfunction, optimizer = "rmsprop", metrics = ["accuracy"])
        history = model.fit(
                X_train, y_train, epochs = 1, batch_size = 4,
                # TODO: if validation set is not empty, the accuracy on validation set will be all zero. Why?
                # TODO: if validation set is empty, it will train 2 examples intead of 1 per epoch.
                callbacks = []
            ) # Initial training of 1 samples per class
        for i in range(epochs):
            # Random strategy
            selected_class = None
            if strat == "random":
                selected_class = random.choice(CLASSES)
            if strat == "variance":
                pass             # TODO: Implement Variance, Entropy Strategies
            if strat == "entropy": 
                pass

            selected_index = CLASSES.index(selected_class)
            print("Epoch ", i, "/", epochs, ", random class: ", selected_class)
            X_base, y_base = manager.get_raw_with_classes([selected_class], 1, seed = 42) # Get 40 samples of the class
            y_base = y_base[:,[selected_index]] # Get the corresponding label
            history = model.fit(
                X_train, y_train, epochs = 1, batch_size = 4,
                callbacks = []
            )

        id = str(uuid.uuid4())
        try:
            model.save(save_path + ".h5")
        except:
            print("could not write")

        loss, acc = model.evaluate(X_test, y_test, batch_size= 4)
        print("test_loss", loss)
        print("test_acc", acc)
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
    run_with_strat(get_ConvLSTM, manager, "model_dumps/newstrat", epochs=200)
   

    