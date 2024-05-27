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

def get_validation_set(manager, classes, samples_per_class):
    validation_set = []
    for cls in classes: # Calculate entropy for each class
        X_val_cls, _ = manager.get_raw_with_classes([cls], samples_per_class) # Get one sample from each 
        validation_set.append(X_val_cls)
    return validation_set

def run_leave_one_out(model_creator, manager, save_path, init_samples = 10, init_epochs = 10, iterations=10, validation_size = 4, num_dropouts=10, samples_per_iter = 10, epochs_per_iter = 5, strat="random"):
    confusion = np.zeros((len(CLASSES), len(CLASSES)))

    lossfunction = "categorical_crossentropy"
    logger = Logger(join(LOG_PATH, "leave_on_out.txt"),
        time(),DATASET_ID,"full_dataset",
        model_creator.__name__,
        "for save path see below",
        lossfunction=lossfunction)


    # additional selective training on selected class
    for i, comp_class in enumerate(CLASSES):
    
        base_classes = [x for j,x in enumerate(CLASSES) if j!=i]
        comp_class = CLASSES[i]

        print("Base class: ", base_classes, "Leaving", comp_class, "out.")

        X_base, y_base = manager.get_raw_with_classes([x for j,x in enumerate(CLASSES) if j!=i], init_samples, seed = 42)
        y_base = y_base[:,[j for j in range(len(CLASSES)) if j != i]] 
        X_comp, _ = manager.get_raw_with_classes([comp_class], init_samples, seed = 42)

        input_shape = X_base.shape[1:]
        num_classes = y_base.shape[1]

        model = model_creator(input_shape, num_classes)
        X_train, X_test, y_train, y_test = train_test_split(X_base,y_base, train_size = .8)


        stop_callback = EarlyStopping(monitor = "val_accuracy", patience = 30, mode = "max", restore_best_weights = True)

        model.compile(loss = lossfunction, optimizer = "rmsprop", metrics = ["accuracy"])
        history = model.fit(
            X_train, y_train, epochs = init_epochs, batch_size = 1,
            validation_split = 0.1,
            callbacks = [stop_callback]
        )
        x_val = get_validation_set(manager, base_classes, validation_size)
        for iter in range(iterations) :
            print("Iteration: ", iter, " of ", iterations)
            selected_class = None
            selected_index = -1
            if strat == "random":
                selected_class = random.choice(base_classes)
            
            elif strat == "entropy" or strat == "variance":
                uncertainty = None
                if strat == "variance":
                    uncertainty = calculate_variance_with_dropout(model, x_val, num_dropouts=num_dropouts)
                    print("Variances: ", uncertainty)
                else: 
                    uncertainty = calculate_entropy_with_dropout(model, x_val, num_dropouts=num_dropouts)
                    print("Entropies: ", uncertainty)
                
                selected_index = np.argmax(uncertainty)
            
            selected_class = base_classes[selected_index]
            print("Selected class: ", selected_class)
            X_base, y_base = manager.get_raw_with_classes([selected_class], samples_per_iter, seed = 42)
            y_base = y_base[:,[j for j in range(len(CLASSES)) if j != i]] 
            X_comp, _ = manager.get_raw_with_classes([comp_class], samples_per_iter, seed = 42)

            input_shape = X_base.shape[1:]
            num_classes = y_base.shape[1]

            X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, train_size=0.8)

            stop_callback = EarlyStopping(monitor = "val_accuracy", patience = 30, mode = "max", restore_best_weights = True)

            model.compile(loss = lossfunction, optimizer = "rmsprop", metrics = ["accuracy"])
            history = model.fit(
                X_train, y_train, epochs = epochs_per_iter, batch_size = 1,
                validation_split = 0.1,
                callbacks = [stop_callback]
            )


        id = str(uuid.uuid4())
        try:
            model.save(save_path + ".h5")
        except:
            print("could not write")

        loss, acc = model.evaluate(X_test, y_test, batch_size= 4)
        print("test_loss", loss)
        print("test_acc", acc)

        dist = model.predict(X_comp)
        choice = np.argmax(dist, axis = 1)
        prediction = np.zeros((choice.size, len(base_classes)))
        prediction[np.arange(choice.size), choice] = 1

        row = np.insert(prediction.sum(axis = 0), i, 0)

        confusion[i, :] = row

    print(confusion)

if __name__ == "__main__":
    manager = DatasetManager(DATASET_ID, ["C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"], CLASSES)
    # run_with_less_data(get_ConvLSTM, manager, "model_dumps/less_data/", samples_per_class=1)
    
    # run_on_full_dataset(get_ConvLSTM, manager, "model_dumps/without_corrupted_full_dataset/")
    n_experiments = 10
    epochs = 100
    train_samples_per_class = 40
    validation_samples_per_class = 0
    test_samples_per_class = 40

    run_leave_one_out(get_ConvLSTM, manager, "model_dumps/leave_one_out/", init_epochs=10, init_samples=10, iterations=50, validation_size=10, num_dropouts=10, samples_per_iter=5, epochs_per_iter=10, strat="entropy")