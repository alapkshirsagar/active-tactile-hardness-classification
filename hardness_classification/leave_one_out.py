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

DATASET_ID = "42"
CLASSES = ["hardness" + str(i+1) for i in range(5)]
MODEL_DUMP_PATH = "model_dumps/"
LOG_PATH = "logs/"

def calculate_entropy_with_dropout(model, X_val, num_dropouts=10):
    preds = np.array([model(X_val, training=True) for _ in range(num_dropouts)])
    entropies = -np.mean(preds * np.log(preds + 1e-10), axis=0)
    sum_entropy = np.sum(entropies, axis=0)
    return sum_entropy

def calculate_variance_with_dropout(model, X_val, num_dropouts=10):
    preds = np.array([model(X_val, training=True) for _ in range(num_dropouts)])
    variances = np.var(preds, axis=0)
    mean_variance = np.mean(variances, axis=0)
    return mean_variance

def run_leave_one_out(model_creator, manager, save_path, init_epochs=10, iterations=10, num_dropouts=10,  epochs_per_iter=5, strat="random", augment_factor=10):

    # Ensure memory growth is enabled
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    confusion = np.zeros((len(CLASSES), len(CLASSES)))

    lossfunction = "categorical_crossentropy"
    lossfunction = "categorical_crossentropy"
    logger = Logger(join(LOG_PATH, "leave_on_out.txt"), time(), DATASET_ID, "full_dataset", model_creator.__name__, "for save path see below", lossfunction=lossfunction)
    ground_truth = []
    predictions = []

    for i, comp_class in enumerate(CLASSES):
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

        history = model.fit(X_train, y_train, epochs=init_epochs, validation_split=0.1, callbacks=[stop_callback])
        for epoch in range(init_epochs):
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
            
            X_train = X_base_iter
            y_train = y_base_iter
            # X_train = np.concatenate([X_train, X_base_iter]) TODO: check if this is necessary
            # y_train = np.concatenate([y_train, y_base_iter]) TODO: check if this is necessary


            model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])
            history = model.fit(X_train, y_train, epochs=epochs_per_iter, batch_size=1, validation_split=0.1, callbacks=[stop_callback])

            for epoch in range(epochs_per_iter):
                run.log({
                    "train_accuracy": history.history["accuracy"][epoch], 
                    "val_accuracy": history.history["val_accuracy"][epoch], 
                    "train_loss": history.history["loss"][epoch], 
                    "val_loss": history.history["val_loss"][epoch]}) # log training and validation accuracy and loss for each epoch

            avg_entropy =np.mean(calculate_entropy_with_dropout(model, X_comp, num_dropouts=num_dropouts)) 
            avg_var = np.mean(calculate_variance_with_dropout(model, X_comp, num_dropouts=num_dropouts))
            
            run.log({"avg_entropy": avg_entropy, "avg_variance": avg_var}) # log average entropy and variance

            # log test accuracy per iteration
            y_preds = model.predict(X_comp)
            y_preds = np.insert(y_preds, i, 0, axis=1) # insert the class that was left out
            preds = np.argmax(y_preds, axis=1) # predicted label
            gts = np.argmax(y_comp, axis=1) # ground truth
            acc = compute_acc(preds, gts, hardness_levels)
            print("Iteration ", i + 1, " Test accuracy: ", acc)
            run.log({"test_accuracy": acc})


        model_save_path = f"{save_path}{uuid.uuid4()}.h5"
        try:
            model.save(model_save_path)
        except Exception as e:
            print("Could not write model:", e)




        dist = model.predict(X_comp)
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
    run.finish()

## Function to return the closest class to a given class
def closest(ground_truth, hardness_lv):
    closest = None
    min_diff = float("inf")
    for i in range(len(hardness_lv)):
        if(i == ground_truth): continue

        diff = abs(hardness_lv[ground_truth] - hardness_lv[i])
        if diff < min_diff:
            min_diff = diff
            closest = i
    return closest

## Function to compute accuracy
def compute_acc(prediction, ground_truth, hardness_lv):
    match = 0
    for(pred, gt) in zip(prediction, ground_truth):
        if(closest(pred, hardness_lv) == closest(gt, hardness_lv)):
            match += 1
    return match/len(ground_truth)


hardness_levels = [0.44, 0.68, 0.79, 1.02, 1.13]


if __name__ == "__main__":
    for i in range(5):
        print("Deform level ", i, "closest: ", closest(i, hardness_levels))
    manager = DatasetManager(DATASET_ID, ["C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"], CLASSES)
    run_leave_one_out(get_ConvLSTM, manager, MODEL_DUMP_PATH, init_epochs=10, iterations=10, num_dropouts=10, epochs_per_iter=100, strat="entropy")
