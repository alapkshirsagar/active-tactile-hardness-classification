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
import tensorflow as tf


DATASET_ID = "42"
CLASSES = ["hardness" + str(i + 1) for i in range(5)]
MODEL_DUMP_PATH = "model_dumps/"
LOG_PATH = "logs/"

def calculate_entropy_with_dropout(model, X_val, num_dropouts=10):
    preds = np.array([model(X_val, training=True) for _ in range(num_dropouts)])
    
    # preds shape: (num_dropouts, num_samples, num_classes)

    # calculate the entropy expectation of each sample over the number of dropouts
    entropies = -np.mean(preds * np.log(preds + 1e-10), axis=0)
    
    # entropies shape: (num_samples, num_classes)

    # Sum the entropies over the examples
    sum_entropy = np.sum(entropies, axis=0)

    # sum_entropy shape: (num_classes,)
    
    return sum_entropy

def calculate_variance_with_dropout(model, X_val, num_dropouts=10):
    preds = np.array([model(X_val, training=True) for _ in range(num_dropouts)])

    # preds shape: (num_dropouts, num_samples, num_classes)
    
    # calculate the variance of each sample over the number of dropouts
    variances = np.var(preds, axis=0)

    # variances shape: (num_samples, num_classes)
    
    # Average the variances over the number of samples
    mean_variance = np.mean(variances, axis=0)

    # mean_variance shape: (num_classes,)
    
    return mean_variance

# train on 5 classes, test on 1
def train_5_test_1(model_creator, manager, save_path, epochs=200, validation_size=4, num_dropouts=10, strat="random"):
    overall_confusion = np.zeros((len(CLASSES), len(CLASSES)))

    lossfunction = "categorical_crossentropy"
    logger = Logger(join(LOG_PATH, "train_5_test_1.txt"),
                    time(), DATASET_ID, "full_dataset",
                    model_creator.__name__,
                    "for save path see below",
                    lossfunction=lossfunction)
    
    # Create validation set, validation_size samples per class
    validation_set = {} # Dictionary to store validation set
    if strat == "entropy" or strat == "variance":
        for cls in CLASSES: # Calculate entropy for each class
            X_val_cls, y_val_cls = manager.get_raw_with_classes([cls], validation_size) # Get one sample from each class
            y_val_cls = y_val_cls[:,[j for j in range(len(CLASSES))]] 
            validation_set[cls] = (X_val_cls, y_val_cls)

    
    X_train, y_train = manager.get_raw_with_classes(CLASSES, 1, seed=42)
    y_train = y_train[:,[j for j in range(len(CLASSES))]]

    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    model = model_creator(input_shape, num_classes)
    # model.summary()

    model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])

    # initial training with 1 sample per class
    model.fit(X_train, y_train, epochs=1, batch_size=4, callbacks=[])

    

    # concatenate the validation sets of training classes, 
    # to ensure that the validation set only contains samples from the training classes 
    # and have the same number of examples over each training class to avoid sampling bias

    X_val = np.concatenate([validation_set[cls][0] for cls in CLASSES])

    # further training with random samples, one example per epoch

    for epoch in range(epochs):
        selected_class = None 
        if strat == "random":
            selected_class = random.choice(CLASSES)
        
        elif strat == "entropy" or strat == "variance":
            uncertainty = None
            if strat == "variance":
                uncertainty = calculate_variance_with_dropout(model, X_val, num_dropouts=num_dropouts)
                print("Variances: ", uncertainty)
            else: 
                uncertainty = calculate_entropy_with_dropout(model, X_val, num_dropouts=num_dropouts)
                print("Entropies: ", uncertainty)
            
            
            # Select the class with the highest entropy
            selected_index = np.argmax(uncertainty)
            selected_class = CLASSES[selected_index]

        print("Epoch ", epoch + 1, "/", epochs, ", selected class: ", selected_class)
        
        # Add an example from the selected class to the training set
        X_train_epoch, y_train_epoch = manager.get_raw_with_classes([selected_class], 1, seed=42)
        y_train_epoch = y_train_epoch[:,[j for j in range(len(CLASSES))]]

        # Concatenate the new example to the existing training set
        X_train = np.concatenate([X_train, X_train_epoch])
        y_train = np.concatenate([y_train, y_train_epoch])

        # Train the model on the entire training set
        model.fit(X_train, y_train, epochs=1, callbacks=[])


    model_id = str(uuid.uuid4())
    try:
        model_save_path = f"{save_path}_{model_id}.h5"
        model.save(model_save_path)
    except:
        print("could not write")

    X_test, y_test = manager.get_raw_with_classes(CLASSES, 10, seed=42)

    # Adjust y_test to reflect all classes
    y_test = y_test[:,[j for j in range(len(CLASSES))]]
    dist = model.predict(X_test)
    # print prediction accuracy
    print("Prediction accuracy: ", np.mean(np.argmax(dist, axis=1) == np.argmax(y_test, axis=1)))
    

# train on 4 classes, test on the rest 1
def train_4_test_1(model_creator, manager, save_path, iterations=20,validation_size = 4, num_dropouts=10,  strat="random"):
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
    
        X_train, y_train = manager.get_raw_with_classes(training_classes, 5, seed=42)

        # Ensure y_train covers all classes and adjust labels
        y_train_full = np.zeros((y_train.shape[0], len(CLASSES))) # Adjust y_train to reflect all classes
        for i, c in enumerate(training_classes):
            class_idx = CLASSES.index(c)
            y_train_full[:, class_idx] = y_train[:, i] # Set the index of the class to the value of y_train
        y_train = y_train_full

        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]

        model = model_creator(input_shape, num_classes)
        # model.summary()

        model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])

        # initial training with 1 sample per class
        model.fit(X_train, y_train, epochs=1, batch_size=4, callbacks=[])

        

        # concatenate the validation sets of training classes, 
        # to ensure that the validation set only contains samples from the training classes 
        # and have the same number of examples over each training class to avoid sampling bias
        if(strat == "entropy" or strat == "variance"):
            X_val = np.concatenate([validation_set[cls][0] for cls in training_classes])

        # further training with random samples, one example per epoch

        for epoch in range(iterations):
            selected_class = None 
            if strat == "random":
                selected_class = random.choice(training_classes)
            
            elif strat == "entropy" or strat == "variance":
                uncertainty = None
                if strat == "variance":
                    uncertainty = calculate_variance_with_dropout(model, X_val, num_dropouts=num_dropouts)
                    print("Variances: ", uncertainty)
                else: 
                    uncertainty = calculate_entropy_with_dropout(model, X_val, num_dropouts=num_dropouts)
                    print("Entropies: ", uncertainty)
                
                selected_index = -1
                # Select the class with the highest entropy, but not the test class
                for i, _ in enumerate(CLASSES):
                    if CLASSES[i] != test_class and (selected_index == -1 or uncertainty[i] > uncertainty[selected_index]):
                        selected_index = i
                selected_class = CLASSES[selected_index]

            print("Epoch ", epoch + 1, "/", iterations, ", selected class: ", selected_class)
            
            # Add examples from the selected class to the training set
            X_train, y_train = manager.get_raw_with_classes([selected_class], 10, seed=42)
            # Ensure y_train covers all classes and adjust labels
            y_train_full = np.zeros((y_train.shape[0], len(CLASSES))) # Adjust y_train to reflect all classes
            for i, c in enumerate(training_classes):
                class_idx = CLASSES.index(c)
                y_train_full[:, class_idx] = y_train[:, i] # Set the index of the class to the value of y_train
            y_train = y_train_full

            # Concatenate the new example to the existing training set
            # X_train = np.concatenate([X_train, X_train_epoch])
            # y_train = np.concatenate([y_train, y_train_epoch])

            # Train the model on the entire training set
            model.fit(X_train, y_train, epochs=5, callbacks=[])


        model_id = str(uuid.uuid4())
        try:
            model_save_path = f"{save_path}_{model_id}.h5"
            model.save(model_save_path)
        except:
            print("could not write")

        X_test, y_test = manager.get_raw_with_classes([test_class], 20, seed=42)

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
    train_4_test_1(get_ConvLSTM, manager, "model_dumps/newstrat", n_experiments=1, iterations=40, validation_size=10, num_dropouts=5, strat="random")
