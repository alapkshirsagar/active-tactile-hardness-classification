from preprocessing import *
from models import *
from log import Logger
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np
import random
from time import time
import uuid
import pandas as pd
from utils import *
import wandb
import sklearn as sk
import argparse

DATASET_ID = "42"
CLASSES = ["hardness" + str(i+1) for i in range(5)]
MODEL_DUMP_PATH = "model_dumps/"
LOG_PATH = "logs/"

## Function to remove columns from a 2D array
# @param y: 2D array to remove columns from
# @param indices: List of indices to remove, in this case the comp_classes
def purge_y_columns(y, removedIndices):
    return y[:, [j for j in range(len(y[0])) if j in removedIndices]]

## Function to fill columns of y in a 2D array with zeros
def expand_y_columns(y, y_classes,  full_classes) -> np.ndarray: 
    array = np.zeros((len(y), len(full_classes)))
    y_indices = [full_classes.index(c) for c in y_classes]
    array[:, y_indices] = y
    return array

def subsample(max_samples, X_base, y_base): 
    # convert max_samples to int
    max_samples = int(max_samples)
    if X_base.shape[0] > max_samples:
        indices = np.random.choice(X_base.shape[0], max_samples, replace=False)
        X_train = X_base[indices]
        y_train = y_base[indices]
    else:
        X_train, y_train = X_base, y_base
    return X_train, y_train

# wrapper function to predict with dropout enabled and batch size 1
# model(x, training=True) does not support setting the batch size
def predict_with_dropout_and_batch_size_1(model, x):
    # List to store predictions
    all_predictions = []
    
    # Iterate through the input data one sample at a time
    for sample in x:
        # Perform a forward pass with dropout enabled
        prediction = model(tf.expand_dims(sample, axis=0), training=True)
        all_predictions.append(prediction)

    # Combine all predictions into one array
    all_predictions = tf.concat(all_predictions, axis=0)
    
    return all_predictions

def calculate_entropy_with_dropout(model, X_val, num_dropouts=10) -> np.ndarray:
    preds = np.array([predict_with_dropout_and_batch_size_1(model, X_val) for _ in range(num_dropouts)])
    entropies = -np.mean(preds * np.log(preds + 1e-10), axis=0)
    sum_entropy = np.sum(entropies, axis=0)
    return sum_entropy

def calculate_variance_with_dropout(model, X_val, num_dropouts=10) -> np.ndarray:
    preds = np.array([predict_with_dropout_and_batch_size_1(model, X_val) for _ in range(num_dropouts)])
    variances = np.var(preds, axis=0)
    mean_variance = np.mean(variances, axis=0)
    return mean_variance

def acc_on_comp(model, X_comp, base_classes, hardness_levels, comp_index):
    dist = model.predict(X_comp, batch_size=1, verbose=0)
    choice = np.argmax(dist, axis = 1)
    prediction = np.zeros((choice.size, len(base_classes)))
    prediction[np.arange(choice.size), choice] = 1
    ground_truth = np.full(choice.shape, comp_index) # ground truth is the comp_index as integers
    return prediction, compute_acc(choice, ground_truth, hardness_levels)

# Function to try to purge the GPU memory
# if the model is not using the GPU memory, do nothing
def try_purge_GPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) # set memory growth to avoid memory errors
        tf.keras.backend.clear_session() # clear the session to free up memory

def adaptive_sampling(CLASSES, model_creator, manager, save_path, base_classes, comp_classes, init_epochs=10, iterations=10, num_dropouts=10, epochs_per_iter=10, strat="random", augment_factor=10, max_samples=50, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=10, comp_size=1, GPU = True, verbose = 0) -> None:

    print("# Initial Training")
    print(f"{init_sample_per_class} samples per class, {val_per_class} validation samples per class, augment factor as {augment_factor} and {init_epochs} epochs")
    if GPU:
        try_purge_GPU()
    else: tf.config.set_visible_devices([], 'GPU')  # disable GPU
    lossfunction = "categorical_crossentropy"

    run = wandb.init(project="leave_one_out", name= f"{strat}_leave_{comp_classes}_out",
                                            config={"dataset_id": DATASET_ID, 
                                                    "model": model_creator.__name__, 
                                                    "lossfunction": lossfunction,
                                                    "epochs": init_epochs + iterations * epochs_per_iter,
                                                    })

    print("Base class: ", base_classes, "Comp class: ", comp_classes)
    run.log({"base_classes": base_classes, "comp_class": comp_classes})


    X_base, y_base = manager.get_raw_with_classes(base_classes, init_sample_per_class + val_per_class)
    X_base, y_base = manager.augment_data(X_base, y_base, augment_factor=augment_factor) # X_base (4, 5, 240, 320, 3) Y_base (4, 5)
    # convert to np.array
    X_base = np.array(X_base)
    y_base = np.array(y_base)
    # convert Y-base to one hot encoding
    y_base = manager.encoder.transform(y_base)
    # remove the class that are not in the base classes in y_base
    base_classes_indices = [CLASSES.index(c) for c in base_classes]
    y_base = purge_y_columns(y_base, base_classes_indices)

    X_comp, y_comp = manager.get_raw_with_classes(comp_classes, comp_size)
    X_comp, y_comp = manager.augment_data(X_comp, y_comp, augment_factor=augment_factor) # X_comp (4, 5, 240, 320, 3) Y_comp (4, 5)
    X_comp = np.array(X_comp)
    y_comp = np.array(y_comp)


    input_shape = X_base.shape[1:]
    num_classes = y_base.shape[1]

    model = model_creator(input_shape, num_classes)
    X_train, y_train = subsample(max_samples * (val_per_class + init_sample_per_class) / init_sample_per_class, X_base, y_base)


    stop_callback = EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True)
    model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=init_epochs, validation_split=val_per_class/(val_per_class + init_sample_per_class), batch_size=1, callbacks=[], verbose=verbose)
    executed_epochs = len(history.history["accuracy"])
    for epoch in range(executed_epochs):
        if(val_per_class > 0):
            run.log({"train_accuracy": history.history["accuracy"][epoch], 
                    "val_accuracy": history.history["val_accuracy"][epoch], 
                    "train_loss": history.history["loss"][epoch], 
                    "val_loss": history.history["val_loss"][epoch]})
        else:
            run.log({"train_accuracy": history.history["accuracy"][epoch], 
                    "train_loss": history.history["loss"][epoch]})

    x_test_model, y_test_model = manager.get_raw_with_classes(base_classes, comp_size)
    # convert y_test to one hot encoding
    y_test_model = manager.encoder.transform(y_test_model)
    y_test_model = purge_y_columns(y_test_model, base_classes_indices)

    # Evaluation after initial training

    print ("### Eval After Initial Training")
    print("Base classes: "+ ",".join(str(x) for x in base_classes))
    print( "Comp classes: "+ ",".join(str(x) for x in comp_classes))
    y_pred_model = model.predict(x_test_model, batch_size=1, verbose=verbose)

    print("Base classes: "+ ",".join(str(x) for x in base_classes))
    print( "Comp classes: "+ ",".join(str(x) for x in comp_classes))
    y_pred_model = model.predict(x_test_model, batch_size=1, verbose=verbose)

    # Accuracy on base classes
    acc_base = np.sum(np.argmax(y_pred_model, axis=1) == np.argmax(y_test_model, axis=1)) / len(y_test_model) # test accuracy
    print("model test accuracy on base classes: ", acc_base)
    run.log({"acc_base": acc_base})

    # Avg. absolute  in hardness levels on base classes
    abs_diff_base =  hardness_abs_diff(np.argmax(y_pred_model, axis=1), np.argmax(y_test_model, axis=1), hardness_levels)
    print("Avg. absolute difference on base classes: ", abs_diff_base)
    run.log({"abs_diff_base": abs_diff_base})

    # Accuracy on comp class. Prediction is correct if the predicted class is the closest to the ground truth class in terms of hardness level
    pred_comp, acc_comp = acc_on_comp(model, X_comp, base_classes, hardness_levels, CLASSES.index(comp_classes[0]))
    print("Iteration accuracy on comp class: ", acc_comp)
    run.log({"acc_comp": acc_comp})

    # Avg. absolute difference in hardness levels on comp class
    abs_diff_comp = hardness_abs_diff(np.argmax(pred_comp, axis=1), np.argmax(y_comp, axis=1), hardness_levels)
    print("Avg. absolute difference in hardness levels on comp class: ", abs_diff_comp)
    run.log({"abs_diff_comp": abs_diff_comp})
    if(strat == "YOTO"):
        print("Strategy is YOTO, no further iterations")
        model_confusion_matrix = sk.metrics.confusion_matrix(np.argmax(y_test_model, axis=1), np.argmax(y_pred_model, axis=1))
        print("Confusion matrix: \n", model_confusion_matrix)
        run.log({"model_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=np.argmax(y_test_model, axis=1), preds=np.argmax(y_pred_model, axis=1),
                        class_names=base_classes)}) 
        run.finish()
        return model, model_confusion_matrix
    print("## Adaptive Sampling")
    print(f"{strat} strategy, {sample_per_iter_per_class} samples per iteration per class, {val_per_class} validation samples per class and {epochs_per_iter} epochs per iteration")
    for iter in range(iterations):
        if GPU:
            try_purge_GPU()
        print("## Iteration: ", iter + 1, " of ", iterations)

        # Adaptive sampling part

        entr = calculate_entropy_with_dropout(model, X_comp, num_dropouts=num_dropouts)
        var = calculate_variance_with_dropout(model, X_comp, num_dropouts=num_dropouts)
        run.log({"avg_entropy": np.mean(entr) , "avg_variance": np.mean(var)}) # log average entropy and variance
        if strat == "random":
            selected_class = random.choice(base_classes)
        elif strat in ["entropy", "variance"]:
            uncertainty = var if strat == "variance" else entr
            print(f"{strat.capitalize()} values: ", uncertainty)
            selected_index = np.argmax(uncertainty)
            selected_class = base_classes[selected_index]

        print("Selected class: ", selected_class)

        # Reservoir sampling part

        X_base_iter, y_base_iter = manager.get_raw_with_classes([selected_class], sample_per_iter_per_class + val_per_class) # get one sample of the selected class
        X_base_iter, y_base_iter = manager.augment_data(X_base_iter, y_base_iter, augment_factor=augment_factor) # augment the sample to get 10 samples
        X_base_iter = np.array(X_base_iter)
        y_base_iter = np.array(y_base_iter) # convert to np.array
        y_base_iter = purge_y_columns(y_base_iter, base_classes_indices)
        
        X_base = np.append(X_base, X_base_iter, axis=0) # append the new samples to the sampling reservoir
        y_base = np.append(y_base, y_base_iter, axis=0)
        X_train, y_train = subsample(max_samples * (val_per_class + sample_per_iter_per_class) / sample_per_iter_per_class, X_base, y_base)
        print("Reservoir size: ", X_base.shape[0], "Training size: ", X_train.shape[0])
        run.log({"reservoir_size": X_base.shape[0], "training_size": X_train.shape[0]})

        model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])
        stop_callback = EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True)
        
        # Training iteration part

        history = model.fit(X_train, y_train, epochs=epochs_per_iter, batch_size=1, validation_split=val_per_class/(val_per_class + sample_per_iter_per_class), callbacks=[], verbose=verbose)

        executed_epochs = len(history.history["accuracy"])
        for epoch in range(executed_epochs):
            if(val_per_class > 0):
                run.log({
                "train_accuracy": history.history["accuracy"][epoch], 
                "val_accuracy": history.history["val_accuracy"][epoch], 
                "train_loss": history.history["loss"][epoch], 
                "val_loss": history.history["val_loss"][epoch]}) # log training and validation accuracy and loss for each epoch
            else:
                run.log({
                "train_accuracy": history.history["accuracy"][epoch], 
                "train_loss": history.history["loss"][epoch]
                })

        
        # Iteration Evaluation part
        
        print ("### Eval After iteration ", iter + 1)
        print("Base classes: "+ ",".join(str(x) for x in base_classes))
        print( "Comp classes: "+ ",".join(str(x) for x in comp_classes))
        y_pred_model = model.predict(x_test_model, batch_size=1, verbose=verbose)

        # Accuracy on base classes
        acc_base = np.sum(np.argmax(y_pred_model, axis=1) == np.argmax(y_test_model, axis=1)) / len(y_test_model) # test accuracy
        print("model test accuracy on base classes: ", acc_base)
        run.log({"acc_base": acc_base})

        # Avg. absolute  in hardness levels on base classes
        abs_diff_base =  hardness_abs_diff(np.argmax(y_pred_model, axis=1), np.argmax(y_test_model, axis=1), hardness_levels)
        print("Avg. absolute difference on base classes: ", abs_diff_base)
        run.log({"abs_diff_base": abs_diff_base})

        # Accuracy on comp class. Prediction is correct if the predicted class is the closest to the ground truth class in terms of hardness level
        pred_comp, acc_comp = acc_on_comp(model, X_comp, base_classes, hardness_levels, CLASSES.index(comp_classes[0]))
        print("Iteration accuracy on comp class: ", acc_comp)
        run.log({"acc_comp": acc_comp})

        # Avg. absolute difference in hardness levels on comp class
        abs_diff_comp = hardness_abs_diff(np.argmax(pred_comp, axis=1), np.argmax(y_comp, axis=1), hardness_levels)
        print("Avg. absolute difference in hardness levels on comp class: ", abs_diff_comp)
        run.log({"abs_diff_comp": abs_diff_comp})


        model_confusion_matrix = sk.metrics.confusion_matrix(np.argmax(y_test_model, axis=1), np.argmax(y_pred_model, axis=1))
        print("Confusion matrix: \n", model_confusion_matrix)
        run.log({"model_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=np.argmax(y_test_model, axis=1), preds=np.argmax(y_pred_model, axis=1),
                        class_names=base_classes)})  


    run.finish()
    return model, model_confusion_matrix

## Function to return the closest class to a given class
def closest(ground_truth, hardness_lv):
    
    # Calculate the absolute differences
    differences = np.abs(hardness_lv - hardness_lv[ground_truth])
    
    # Set the difference for the ground_truth index to infinity to exclude it
    differences[ground_truth] = np.inf
    
    # Find the index of the minimum difference
    closest_class = np.argmin(differences)
    
    return closest_class


## Function to compute accuracy
def compute_acc(prediction, ground_truth, hardness_lv):
    match = 0
    for(pred, gt) in zip(prediction, ground_truth):
        if(pred == closest(gt, hardness_lv)):
            match += 1
    return match/len(ground_truth)


hardness_levels = np.array([1.13, 1.02, 0.79, 0.68, 0.44]) 

## Function to compute the avg. absolute difference in hardness levels
def hardness_abs_diff(prediction, ground_truth, hardness_lv): 
    diff = 0
    for(pred, gt) in zip(prediction, ground_truth):
        diff += np.abs(hardness_lv[pred] - hardness_lv[gt])
    return diff/len(ground_truth)

if __name__ == "__main__":
    def train_model_A(base_cls, test_cls, strat="variance", verbose=0):
            return adaptive_sampling(CLASSES, get_ConvLSTM, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=10, iterations=10, num_dropouts=10, epochs_per_iter=10, strat=strat,augment_factor=1, max_samples=40, init_sample_per_class=4, sample_per_iter_per_class=1, val_per_class=0, comp_size=10, GPU=True, verbose=verbose)

    def train_model_debug(base_cls, test_cls):
            return adaptive_sampling(CLASSES, get_ConvLSTM, manager, MODEL_DUMP_PATH, base_cls, test_cls, init_epochs=1, iterations=1, num_dropouts=1, epochs_per_iter=1, strat="variance",augment_factor=1, max_samples=40, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=0, comp_size=1, GPU=True)

    for i in range(5):
        print("Deform level ", i, "closest: ", closest(i, hardness_levels))
    # Relative Path
    manager = DatasetManager(["../../IP-Object-Hardness/data_no_noise_vel_02to07_force_1to5/data"], classes=CLASSES, seq_length=4)
    # Windows labsolute path "C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"
    # Since the dataset has been processed and cached in the working directory, we can load it directly
    # Already removed the dataset to elsewhere, if the cache is not found, it will raise an error and need to be moved back

    parser = argparse.ArgumentParser(description='Adaptive Sampling Script')
    parser.add_argument('--i', type=int, default=0, help='idx of the class to leave out')
    parser.add_argument('--strat', type=str, default="variance", choices=['random', 'entropy', 'variance', 'YOTO'], help='Sampling strategy')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level. 0: silent, 1: progress bar, 2: one line per epoch')

    args = parser.parse_args()
    i = args.i
    strat = args.strat
    verbose = args.verbose
    print(f"# leave {CLASSES[i]} out, strategy: {strat}")
    model, _ = train_model_A(base_cls=[CLASSES[j] for j in range(5) if j != i], test_cls=[CLASSES[i]], strat=strat, verbose=verbose)
    model.save("./AdaptiveModels/A1/leave_" + CLASSES[i] + "_out.h5")


    # save model


