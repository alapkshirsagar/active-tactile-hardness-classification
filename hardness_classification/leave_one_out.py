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


def calculate_entropy_with_dropout(model, X_val, num_dropouts=10) -> np.ndarray:
    preds = np.array([model(X_val, training=True) for _ in range(num_dropouts)])
    entropies = -np.mean(preds * np.log(preds + 1e-10), axis=0)
    sum_entropy = np.sum(entropies, axis=0)
    return sum_entropy

def calculate_variance_with_dropout(model, X_val, num_dropouts=10) -> np.ndarray:
    preds = np.array([model(X_val, training=True) for _ in range(num_dropouts)])
    variances = np.var(preds, axis=0)
    mean_variance = np.mean(variances, axis=0)
    return mean_variance

# Function to try to purge the GPU memory
# if the model is not using the GPU memory, do nothing
def try_purge_GPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) # set memory growth to avoid memory errors
        tf.keras.backend.clear_session() # clear the session to free up memory

def adaptive_sampling(CLASSES, model_creator, manager, save_path, base_classes, comp_classes, init_epochs=10, iterations=10, num_dropouts=10, epochs_per_iter=10, strat="random", augment_factor=10, max_samples=50, init_sample_per_class=1, sample_per_iter_per_class=1, val_per_class=10, comp_size=1, GPU = True) -> None:
    if GPU:
        try_purge_GPU()
    else: tf.config.set_visible_devices([], 'GPU')  # disable GPU
    lossfunction = "categorical_crossentropy"

    run = wandb.init(project="leave_one_out", name= f"leave_{comp_classes}_out",
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

    history = model.fit(X_train, y_train, epochs=init_epochs, validation_split=val_per_class/(val_per_class + init_sample_per_class), batch_size=1, callbacks=[])
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

    for iter in range(iterations):
        if GPU:
            try_purge_GPU()
        print("Iteration: ", iter, " of ", iterations)
    
        if strat == "random":
            selected_class = random.choice(base_classes)
        elif strat in ["entropy", "variance"]:
            uncertainty = calculate_variance_with_dropout(model, X_comp, num_dropouts=num_dropouts) if strat == "variance" else calculate_entropy_with_dropout(model, X_comp, num_dropouts=num_dropouts)
            print(f"{strat.capitalize()} values: ", uncertainty)
            selected_index = np.argmax(uncertainty)
            selected_class = base_classes[selected_index]

        print("Selected class: ", selected_class)

        X_base_iter, y_base_iter = manager.get_raw_with_classes([selected_class], sample_per_iter_per_class + val_per_class) # get one sample of the selected class
        X_base_iter, y_base_iter = manager.augment_data(X_base_iter, y_base_iter, augment_factor=augment_factor) # augment the sample to get 10 samples
        X_base_iter = np.array(X_base_iter)
        y_base_iter = np.array(y_base_iter) # convert to np.array
        y_base_iter = purge_y_columns(y_base_iter, base_classes_indices)
        
        # append the new samples to the sampling reservoir
        X_base = np.append(X_base, X_base_iter, axis=0)
        y_base = np.append(y_base, y_base_iter, axis=0)

        # Subsampling: if the number of samples exceeds max_samples, randomly select max_samples samples
        X_train, y_train = subsample(max_samples * (val_per_class + sample_per_iter_per_class) / init_sample_per_class, X_base, y_base)

        print("Reservoir size: ", X_base.shape[0], "Training size: ", X_train.shape[0])

        model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])
        stop_callback = EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True)

        history = model.fit(X_train, y_train, epochs=epochs_per_iter, batch_size=1, validation_split=val_per_class/(val_per_class + sample_per_iter_per_class), callbacks=[])

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
        avg_entropy =np.mean(calculate_entropy_with_dropout(model, X_comp, num_dropouts=num_dropouts)) 
        avg_var = np.mean(calculate_variance_with_dropout(model, X_comp, num_dropouts=num_dropouts))
        
        run.log({"avg_entropy": avg_entropy, "avg_variance": avg_var}) # log average entropy and variance

        # log test accuracy per iteration
        y_preds = model.predict(X_comp, batch_size=1)

        # insert the comp_classes that were left out
        y_preds = expand_y_columns(y_preds, base_classes, CLASSES)
        
        preds = np.argmax(y_preds, axis=1) # predicted label
        gts = np.argmax(y_comp, axis=1) # ground truth
        acc = compute_acc(preds, gts, hardness_levels)
        print("Iteration ", iter + 1, " Test accuracy: ", acc)
        run.log({"test_accuracy": acc})

    # evaluate the model

    x_test_model, y_test_model = manager.get_raw_with_classes(base_classes, 10)
    y_test_model = purge_y_columns(y_test_model, base_classes_indices)

    y_pred_model = model.predict(x_test_model, batch_size=1)

    acc = np.sum(np.argmax(y_pred_model, axis=1) == np.argmax(y_test_model, axis=1)) / len(y_test_model) # test accuracy
    print("Base classes: "+ ",".join(str(x) for x in base_classes))
    print( "Comp classes: "+ ",".join(str(x) for x in comp_classes))
    print("model test accuracy: ", acc)
    run.log({"final_test_accuracy": acc})
    model_confusion_matrix = sk.metrics.confusion_matrix(np.argmax(y_test_model, axis=1), np.argmax(y_pred_model, axis=1))
    print("Confusion matrix: \n", model_confusion_matrix)
    run.log({"model_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                    y_true=np.argmax(y_test_model, axis=1), preds=np.argmax(y_pred_model, axis=1),
                    class_names=base_classes)})

    model_save_path = f"{save_path}{uuid.uuid4()}.h5"
    try:
        model.save(model_save_path)
    except Exception as e:
        print("Could not write model:", e)

    dist = model.predict(X_comp, batch_size=1)
    choice = np.argmax(dist, axis = 1)
    prediction = np.zeros((choice.size, len(base_classes)))
    prediction[np.arange(choice.size), choice] = 1
    run.finish()


def run_leave_one_out(model_creator, manager, save_path, init_epochs=10, iterations=10, num_dropouts=10,  epochs_per_iter=5, strat="random", augment_factor=10, max_samples=50):

    # Ensure memory growth is enabled
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    confusion = np.zeros((len(CLASSES), len(CLASSES)))

    lossfunction = "categorical_crossentropy"
    logger = Logger(join(LOG_PATH, "leave_on_out.txt"), time(), DATASET_ID, "full_dataset", model_creator.__name__, "for save path see below", lossfunction=lossfunction)
    ground_truth = []
    predictions = []

    for i, comp_class in enumerate(CLASSES):
        if gpus:
            tf.keras.backend.clear_session()
        run = wandb.init(project="leave_one_out", name= f"leave_{comp_class}_out",
                                            config={"dataset_id": DATASET_ID, 
                                                    "model": model_creator.__name__, 
                                                    "lossfunction": lossfunction,
                                                    "epochs": init_epochs + iterations * epochs_per_iter,
                                                    })

        base_classes = [x for j, x in enumerate(CLASSES) if j != i]
        print("Base class: ", base_classes, "Leaving", comp_class, "out.")
        run.log({"base_classes": base_classes, "comp_class": comp_class})


        X_base, y_base = manager.get_raw_with_classes(base_classes, 1)
        X_base, y_base = manager.augment_data(X_base, y_base, augment_factor=augment_factor) # X_base (4, 5, 240, 320, 3) Y_base (4, 5)
        # convert to np.array
        X_base = np.array(X_base)
        y_base = np.array(y_base)

        y_base = y_base[:, [j for j in range(len(CLASSES)) if j != i]]
        X_comp, y_comp = manager.get_raw_with_classes([comp_class], 1)
        X_comp, y_comp = manager.augment_data(X_comp, y_comp, augment_factor=augment_factor) # X_comp (4, 5, 240, 320, 3) Y_comp (4, 5)
        X_comp = np.array(X_comp)
        y_comp = np.array(y_comp)


        input_shape = X_base.shape[1:]
        num_classes = y_base.shape[1]

        model = model_creator(input_shape, num_classes)
        X_train, y_train = X_base, y_base

        stop_callback = EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True)
        model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])

        history = model.fit(X_train, y_train, epochs=init_epochs, validation_split=0.1, batch_size=1, callbacks=[])
        executed_epochs = len(history.history["accuracy"])
        for epoch in range(executed_epochs):
            run.log({"train_accuracy": history.history["accuracy"][epoch], 
                       "val_accuracy": history.history["val_accuracy"][epoch], 
                       "train_loss": history.history["loss"][epoch], 
                       "val_loss": history.history["val_loss"][epoch]})

        for iter in range(iterations):
            print("Iteration: ", iter, " of ", iterations)
        
            if strat == "random":
                selected_class = random.choice(base_classes)
            elif strat in ["entropy", "variance"]:
                uncertainty = calculate_variance_with_dropout(model, X_comp, num_dropouts=num_dropouts) if strat == "variance" else calculate_entropy_with_dropout(model, X_comp, num_dropouts=num_dropouts)
                print(f"{strat.capitalize()} values: ", uncertainty)
                selected_index = np.argmax(uncertainty)
                selected_class = base_classes[selected_index]

            print("Selected class: ", selected_class)

            X_base_iter, y_base_iter = manager.get_raw_with_classes([selected_class], 1) # get one sample of the selected class
            X_base_iter, y_base_iter = manager.augment_data(X_base_iter, y_base_iter, augment_factor=augment_factor) # augment the sample to get 10 samples
            X_base_iter = np.array(X_base_iter)
            y_base_iter = np.array(y_base_iter) # convert to np.array
            y_base_iter = y_base_iter[:, [j for j in range(len(CLASSES)) if j != i]]
            
            # append the new samples to the sampling reservoir
            X_base = np.append(X_base, X_base_iter, axis=0)
            y_base = np.append(y_base, y_base_iter, axis=0)

            # Subsampling: if the number of samples exceeds max_samples, randomly select max_samples samples
            if X_base.shape[0] > max_samples:
                indices = np.random.choice(X_train.shape[0], max_samples, replace=False)
                X_train = X_base[indices]
                y_train = y_base[indices]
            else:
                X_train, y_train = X_base, y_base

            print("Reservoir size: ", X_base.shape[0], "Training size: ", X_train.shape[0])

            model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])
            stop_callback = EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True)

            history = model.fit(X_train, y_train, epochs=epochs_per_iter, batch_size=1, validation_split=0.1, callbacks=[])

            executed_epochs = len(history.history["accuracy"])
            for epoch in range(executed_epochs):

                run.log({
                    "train_accuracy": history.history["accuracy"][epoch], 
                    "val_accuracy": history.history["val_accuracy"][epoch], 
                    "train_loss": history.history["loss"][epoch], 
                    "val_loss": history.history["val_loss"][epoch]}) # log training and validation accuracy and loss for each epoch

            avg_entropy =np.mean(calculate_entropy_with_dropout(model, X_comp, num_dropouts=num_dropouts)) 
            avg_var = np.mean(calculate_variance_with_dropout(model, X_comp, num_dropouts=num_dropouts))
            
            run.log({"avg_entropy": avg_entropy, "avg_variance": avg_var}) # log average entropy and variance

            # log test accuracy per iteration
            y_preds = model.predict(X_comp, batch_size=1)
            y_preds = np.insert(y_preds, i, 0, axis=1) # insert the class that was left out
            preds = np.argmax(y_preds, axis=1) # predicted label
            gts = np.argmax(y_comp, axis=1) # ground truth
            acc = compute_acc(preds, gts, hardness_levels)
            print("Iteration ", iter + 1, " Test accuracy: ", acc)
            run.log({"test_accuracy": acc})

        # evaluate the model

        x_test_model, y_test_model = manager.get_raw_with_classes(base_classes, 10)
        y_test_model = y_test_model[:, [j for j in range(len(CLASSES)) if j != i]]

        y_pred_model = model.predict(x_test_model, batch_size=1)

        acc = np.sum(np.argmax(y_pred_model, axis=1) == np.argmax(y_test_model, axis=1)) / len(y_test_model) # test accuracy
        print("leave " + comp_class + " out model test accuracy: ", acc)
        run.log({"final_test_accuracy": acc})
        model_confusion_matrix = sk.metrics.confusion_matrix(np.argmax(y_test_model, axis=1), np.argmax(y_pred_model, axis=1))
        print("Confusion matrix: \n", model_confusion_matrix)
        run.log({"model_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=np.argmax(y_test_model, axis=1), preds=np.argmax(y_pred_model, axis=1),
                        class_names=base_classes)})

        model_save_path = f"{save_path}{uuid.uuid4()}.h5"
        try:
            model.save(model_save_path)
        except Exception as e:
            print("Could not write model:", e)




        dist = model.predict(X_comp, batch_size=1)
        choice = np.argmax(dist, axis = 1)
        prediction = np.zeros((choice.size, len(base_classes)))
        prediction[np.arange(choice.size), choice] = 1

        row = np.insert(prediction.sum(axis = 0), i, 0)
        
        confusion[i, :] = row
        ground_truth.extend(np.argmax(y_comp, axis=1))
    
        predictions.extend(np.argmax(np.insert(dist, i, 0, axis=1), axis=1))

        run.finish()


    print(confusion)
    # log confusion matrix to wandb
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    run = wandb.init(project="leave_one_out", name = "confusion_matrix",
                                            config={"dataset_id": DATASET_ID, 
                                                    "model": model_creator.__name__, 
                                                    "lossfunction": lossfunction,
                                                    "epochs": init_epochs + iterations * epochs_per_iter,
                                                    })
    run.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=ground_truth, preds=predictions,
                        class_names=CLASSES)})
    
    acc = compute_acc(predictions, ground_truth, hardness_levels)
    print("overall_test_acc", acc)
    run.log({"overall_test_accuracy": acc})
    run.log({"Strategy": strat})
    run.finish()

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
        if(closest(pred, hardness_lv) == closest(gt, hardness_lv)):
            match += 1
    return match/len(ground_truth)


hardness_levels = np.array([0.44, 0.68, 0.79, 1.02, 1.13]) 


if __name__ == "__main__":
    for i in range(5):
        print("Deform level ", i, "closest: ", closest(i, hardness_levels))
    manager = DatasetManager(DATASET_ID, ["C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"], CLASSES)

    adaptive_sampling(CLASSES, get_ConvLSTM, manager, MODEL_DUMP_PATH, ["hardness2", "hardness3", "hardness4", "hardness5"], ["hardness1"], init_epochs=100, iterations=10, num_dropouts=10, epochs_per_iter=100, strat="variance", augment_factor=1, max_samples=50, init_sample_per_class=10, sample_per_iter_per_class=1, val_per_class=0, comp_size=1)

    # adaptive_sampling(CLASSES, get_ConvLSTM, manager, MODEL_DUMP_PATH, ["hardness2", "hardness3", "hardness4", "hardness5"], ["hardness1"], init_epochs=1, iterations=5, num_dropouts=10, epochs_per_iter=1, strat="variance", augment_factor=10, max_samples=50, init_sample_per_class=5, sample_per_iter_per_class=1, val_per_class=2, comp_size=1, GPU = True)
