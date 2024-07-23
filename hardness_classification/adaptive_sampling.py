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

DATASET_ID = "42"
# CLASSES = ["hardness" + str(i+1) for i in range(5)]
MODEL_DUMP_PATH = "model_dumps/"
LOG_PATH = "logs/"

__all__ = ["adaptive_sampling", "base_comp_split", "subsample", "predict_with_dropout_and_batch_size_1", "calculate_entropy_with_dropout", "calculate_variance_with_dropout", "acc_on_comp", "try_purge_GPU", "closest", "compute_acc", "hardness_abs_diff", "DATASET_ID", "MODEL_DUMP_PATH", "LOG_PATH"]


def base_comp_split(X, y, base_classes, comp_classes):
    mask_base = np.isin(y, base_classes)
    mask_comp = np.isin(y, comp_classes)
    # Apply masks to split the data
    X_base = X[mask_base]
    y_base = y[mask_base]
    X_comp = X[mask_comp]
    y_comp = y[mask_comp]
    return X_base, y_base, X_comp, y_comp

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
        prediction = model.predict_raw(tf.expand_dims(sample, axis=0), training=True)
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
    prediction = np.zeros((choice.size, len(hardness_levels)))
    prediction[np.arange(choice.size), choice] = 1
    ground_truth = np.full(choice.shape, comp_index) # ground truth is the comp_index as integers
    return prediction, compute_acc(choice, ground_truth, hardness_levels)

# Function to try to purge the GPU memory
# if the model is not using the GPU memory, do nothing
def try_purge_GPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.keras.backend.clear_session() # clear the session to free up memory

def adaptive_sampling(CLASSES, hardness_levels, model, manager, save_path, base_classes, comp_classes, init_epochs=10, iterations=10, num_dropouts=10, epochs_per_iter=10, strat="random", augment_factor=10, max_samples=50, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=10, comp_size=1, GPU = True, verbose = 0, ) -> None:

    def eval_model():
                # Iteration Evaluation part
        
        print("Base classes: "+ ",".join(str(x) for x in base_classes))
        print( "Comp classes: "+ ",".join(str(x) for x in comp_classes))
        y_pred_model = model.predict(x_test_base, batch_size=1, verbose=verbose)
        y_pred_argmax = np.argmax(y_pred_model, axis=1) # predicted classes on base classes
        y_test_argmax = np.argmax(y_test_base, axis=1) # ground truth on base classes
        y_comp_argmax = np.argmax(y_comp, axis=1) # ground truth on comp class

        # Accuracy on base classes
        acc_base = np.sum(y_pred_argmax == y_test_argmax) / len(y_test_base) # test accuracy
        print("model test accuracy on base classes: ", acc_base)
        run.log({"acc_base": acc_base})

        # Avg. absolute  in hardness levels on base classes
        abs_diff_base =  hardness_abs_diff(y_pred_argmax, y_test_argmax, hardness_levels)
        print("Avg. absolute difference on base classes: ", abs_diff_base)
        run.log({"abs_diff_base": abs_diff_base})

        # Accuracy on comp class. Prediction is correct if the predicted class is the closest to the ground truth class in terms of hardness level
        pred_comp, acc_comp = acc_on_comp(model, X_comp, base_classes, hardness_levels, CLASSES.index(comp_classes[0]))
        print("Iteration accuracy on comp class: ", acc_comp)
        run.log({"acc_comp": acc_comp})

        # Avg. absolute difference in hardness levels on comp class
        abs_diff_comp = hardness_abs_diff(np.argmax(pred_comp, axis=1), y_comp_argmax, hardness_levels)
        print("Avg. absolute difference in hardness levels on comp class: ", abs_diff_comp)
        run.log({"abs_diff_comp": abs_diff_comp})

        pred_comp_argmax = np.argmax(pred_comp, axis=1) # prediction on comp class
        pred_all = np.append(y_pred_argmax, pred_comp_argmax) # combine predictions on base and comp classes
        gt_all = np.append(y_test_argmax, y_comp_argmax) # combine ground truth on base and comp classes

        model_confusion_matrix = sk.metrics.confusion_matrix(gt_all, pred_all)
        print("Confusion matrix: \n", model_confusion_matrix)
        run.log({"model_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=gt_all, preds=pred_all,
                        class_names=CLASSES)})  

    print("# Initial Training")
    print(f"{init_sample_per_class} samples per class, {val_per_class} validation samples per class, augment factor as {augment_factor} and {init_epochs} epochs")
    if GPU:
        try_purge_GPU()
    else: tf.config.set_visible_devices([], 'GPU')  # disable GPU
    lossfunction = "categorical_crossentropy"

    run = wandb.init(project="OpticalFlowNN", name=f"{strat}_leave_{comp_classes}_out",
                                            config={"dataset_id": DATASET_ID, 
                                                    "model": model.model_name, 
                                                    "lossfunction": lossfunction,
                                                    "epochs": init_epochs + iterations * epochs_per_iter,
                                                    })

    print("Base class: ", base_classes, "Comp class: ", comp_classes)
    run.log({"base_classes": base_classes, "comp_class": comp_classes})

    X, y = manager.get_raw_with_classes(CLASSES, init_sample_per_class + val_per_class, augment_factor=augment_factor)
    # filter base classes
    X_base, y_base, _, _ = base_comp_split(X, y, base_classes, [])
    y_base = manager.encoder.transform(y_base)

    

    X_train, y_train = subsample(max_samples * (val_per_class + init_sample_per_class) / init_sample_per_class, X_base, y_base)

    model.compile_model(optimizer="rmsprop", loss=lossfunction, metrics=["accuracy"])

    history = model.train_model(X_train, y_train, batch_size=1, epochs=init_epochs, validation_samples_per_class=val_per_class, train_samples_per_class=init_sample_per_class)
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
            
    X_, y_ = manager.get_raw_with_classes(CLASSES, comp_size, augment_factor=augment_factor, shuffle=True)

    x_test_base, y_test_base, X_comp, y_comp = base_comp_split(X_, y_, base_classes, comp_classes)

    # convert y_test to one hot encoding
    y_test_base = manager.encoder.transform(y_test_base)
    y_comp = manager.encoder.transform(y_comp)
    print("### Eval After Initial Training")
    eval_model()

    if(strat == "YOTO"):
        print("Strategy is YOTO, no further iterations")

        run.finish()
        return model
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
            print(f"{strat} values: ", uncertainty)
            selected_index = 0
            max_uncertainty = -1.0
            for i in range(0, len(uncertainty)):
                if uncertainty[i] > max_uncertainty and CLASSES[i] in base_classes:
                    max_uncertainty = uncertainty[i]
                    selected_index = i
            print("Selected index: ", selected_index)
            
            selected_class = CLASSES[selected_index]

        # Reservoir sampling part

        X_base_iter, y_base_iter = manager.get_raw_with_classes(CLASSES, sample_per_iter_per_class + val_per_class, augment_factor=augment_factor,shuffle=True) # get one sample of the selected class
        X_base_iter, y_base_iter, _, _ = base_comp_split(X_base_iter, y_base_iter, [selected_class], [])
        y_base_iter = manager.encoder.transform(y_base_iter)

        X_base = np.append(X_base, X_base_iter, axis=0) # append the new samples to the sampling reservoir
        y_base = np.append(y_base, y_base_iter, axis=0)
        X_train, y_train = subsample(max_samples * (val_per_class + sample_per_iter_per_class) / sample_per_iter_per_class, X_base, y_base)
        print("Reservoir size: ", X_base.shape[0], "Training size: ", X_train.shape[0])
        run.log({"reservoir_size": X_base.shape[0], "training_size": X_train.shape[0]})

        # Training iteration part

        model.compile_model(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])
        stop_callback = EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True)

        history = model.train_model(X_train, y_train, batch_size=1, epochs=epochs_per_iter, validation_samples_per_class=val_per_class, train_samples_per_class=sample_per_iter_per_class)

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
        print ("### Eval After iteration ", iter + 1)
        eval_model()
        model.save_model(MODEL_DUMP_PATH + model.model_name + "_leave_" + str(CLASSES[i]) + "_out.h5")




    run.finish()
    return model

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


# hardness_levels = np.array([1.13, 1.02, 0.79, 0.68, 0.44]) # Hardness levels for the 5 classes
# hardness_levels = np.array([1.13, 1.02, 0.79, 0.68, 0.44]) # Yuan Dataset

## Function to compute the avg. absolute difference in hardness levels
def hardness_abs_diff(prediction, ground_truth, hardness_lv): 
    diff = 0
    for(pred, gt) in zip(prediction, ground_truth):
        diff += np.abs(hardness_lv[pred] - hardness_lv[gt])
    return diff/len(ground_truth)
