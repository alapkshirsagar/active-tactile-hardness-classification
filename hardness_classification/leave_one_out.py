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

def run_leave_one_out(model_creator, manager, save_path, test_samples=10, init_samples=10, init_epochs=10, iterations=10, num_dropouts=10, samples_per_iter=10, epochs_per_iter=5, strat="random"):
    confusion = np.zeros((len(CLASSES), len(CLASSES)))

    lossfunction = "categorical_crossentropy"
    lossfunction = "categorical_crossentropy"
    logger = Logger(join(LOG_PATH, "leave_on_out.txt"), time(), DATASET_ID, "full_dataset", model_creator.__name__, "for save path see below", lossfunction=lossfunction)

    for i, comp_class in enumerate(CLASSES):
        wandb.init(project="leave_one_out", config={"dataset_id": DATASET_ID, 
                                                    "model": model_creator.__name__, 
                                                    "name": f"leave_{comp_class}_out",
                                                    "lossfunction": lossfunction,
                                                    "epochs": init_epochs + iterations * epochs_per_iter,
                                                    })

        base_classes = [x for j, x in enumerate(CLASSES) if j != i]
        print("Base class: ", base_classes, "Leaving", comp_class, "out.")
        wandb.log({"base_classes": base_classes, "comp_class": comp_class})

        X_base, y_base = manager.get_raw_with_classes(base_classes, init_samples, seed=42)
        y_base = y_base[:, [j for j in range(len(CLASSES)) if j != i]]
        X_comp, _ = manager.get_raw_with_classes([comp_class], test_samples, seed=42)

        input_shape = X_base.shape[1:]
        num_classes = y_base.shape[1]

        model = model_creator(input_shape, num_classes)
        X_train, y_train = X_base, y_base

        stop_callback = EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True)
        model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])

        history = model.fit(X_train, y_train, epochs=init_epochs, validation_split=0.1, callbacks=[stop_callback])
        for epoch in range(init_epochs):
            wandb.log({"train_accuracy": history.history["accuracy"][epoch], 
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
            X_base_iter, y_base_iter = manager.get_raw_with_classes([selected_class], samples_per_iter, seed=42)
            y_base_iter = y_base_iter[:, [j for j in range(len(CLASSES)) if j != i]]
            X_train = np.concatenate([X_train, X_base_iter])
            y_train = np.concatenate([y_train, y_base_iter])

            X_train_split, X_test, y_train_split, y_test = train_test_split(X_train, y_train, train_size=0.8)

            model.compile(loss=lossfunction, optimizer="rmsprop", metrics=["accuracy"])
            history = model.fit(X_train_split, y_train_split, epochs=epochs_per_iter, batch_size=1, validation_split=0.1, callbacks=[stop_callback])

            for epoch in range(epochs_per_iter):
                wandb.log({
                    "train_accuracy": history.history["accuracy"][epoch], 
                    "val_accuracy": history.history["val_accuracy"][epoch], 
                    "train_loss": history.history["loss"][epoch], 
                    "val_loss": history.history["val_loss"][epoch]}) # log training and validation accuracy and loss for each epoch

            avg_entropy =np.mean(calculate_entropy_with_dropout(model, X_comp, num_dropouts=num_dropouts)) 
            avg_var = np.mean(calculate_variance_with_dropout(model, X_comp, num_dropouts=num_dropouts))
            wandb.log({"avg_entropy": avg_entropy, "avg_variance": avg_var}) # log average entropy and variance
        model_save_path = f"{save_path}{uuid.uuid4()}.h5"
        try:
            model.save(model_save_path)
        except Exception as e:
            print("Could not write model:", e)

        loss, acc = model.evaluate(X_test, y_test, batch_size=4)
        print("test_loss", loss)
        print("test_acc", acc)
        wandb.log({"test_loss": loss, "test_accuracy": acc})

        dist = model.predict(X_comp)
        choice = np.argmax(dist, axis=1)
        prediction = np.zeros((choice.size, len(base_classes)))
        prediction[np.arange(choice.size), choice] = 1

        row = np.insert(prediction.sum(axis=0), i, 0)
        confusion[i, :] = row

    print(confusion)
    wandb.log({"confusion_matrix": confusion.tolist()})
    wandb.finish()

if __name__ == "__main__":
    manager = DatasetManager(DATASET_ID, ["C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"], CLASSES)
    run_leave_one_out(get_ConvLSTM, manager, MODEL_DUMP_PATH, init_epochs=2, init_samples=1, iterations=4, num_dropouts=10, samples_per_iter=1, epochs_per_iter=10, strat="entropy")
