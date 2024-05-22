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



def run_on_full_dataset(model_creator, manager, save_path):
    lossfunction = "categorical_crossentropy"
    model_save_path = save_path + datetime.datetime.fromtimestamp(time()).strftime('%d-%m-%Y_%H-%M-%S') + ".h5"
    logger = Logger(join(LOG_PATH, "full.txt"),
        time(),DATASET_ID,"full_dataset",
        model_creator.__name__,
        model_save_path,
        lossfunction=lossfunction)

    X,y = manager.get_raw()
    input_shape = X.shape[1:]
    num_classes = y.shape[1]
    model = model_creator(input_shape, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 42)
    logger.log("Number of Datapoints per class" + str(y_train.sum(axis = 0)))

    stop_callback = EarlyStopping(monitor = "val_accuracy", patience = 30, mode = "max", restore_best_weights = True)

    model.compile(loss = lossfunction, optimizer = "rmsprop", metrics = ["accuracy"])
    history = model.fit(
        X_train, y_train, epochs = 100, batch_size = 4,
        validation_split = 0.1,
        callbacks = [stop_callback]
    )
    try:
        model.save(model_save_path)
    except:
        print("could not write")


    results = model.evaluate(X_test, y_test, batch_size= 4)
    logger.log("test_loss, test_acc = " + str(results))
    logger.log("Required epochs:" + str(len(history.history['loss'])))

    y_hat = model.predict(X)
    matrix = confusion_matrix(predictions = np.argmax(y_hat, axis = 1),labels = np.argmax(y, axis = 1))
    logger.log("Confusion Matrix on full data dataset")
    logger.log(matrix)

def run_leave_one_out(model_creator, manager, save_path):
    confusion = np.zeros((len(CLASSES), len(CLASSES)))

    lossfunction = "categorical_crossentropy"
    logger = Logger(join(LOG_PATH, "leave_on_out.txt"),
        time(),DATASET_ID,"full_dataset",
        model_creator.__name__,
        "for save path see below",
        lossfunction=lossfunction)


    for i in range(len(CLASSES)):
    
        base_classes = [x for j,x in enumerate(CLASSES) if j!=i]
        comp_class = CLASSES[i]

        print("Leaving", comp_class, "out.")

        X_base, y_base = manager.get_raw_with_classes([x for j,x in enumerate(CLASSES) if j!=i])
        y_base = y_base[:,[j for j in range(len(CLASSES)) if j != i]] 
        X_comp, _ = manager.get_raw_with_classes([comp_class])

        input_shape = X_base.shape[1:]
        num_classes = y_base.shape[1]

        model = model_creator(input_shape, num_classes)
        model.summary()
        X_train, X_test, y_train, y_test = train_test_split(X_base,y_base, train_size = .8)


        stop_callback = EarlyStopping(monitor = "val_accuracy", patience = 30, mode = "max", restore_best_weights = True)

        model.compile(loss = lossfunction, optimizer = "rmsprop", metrics = ["accuracy"])
        history = model.fit(
            X_train, y_train, epochs = 100, batch_size = 4,
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

def run_train_2_classify_three(model_creator, manager, n_experiments, epochs, train_samples_per_class, validation_samples_per_class, test_samples_per_class, lossfunction = "categorical_crossentropy", save_path=None, load_path=None):
    timestamp = datetime.datetime.fromtimestamp(time()).strftime('%d-%m-%Y_%H-%M-%S')
    if save_path is None and load_path is None:
        raise ValueError("Either save_path or load_path must be specified")
    elif save_path is not None and load_path is not None:
        raise ValueError("Only one of save_path or load_path must be specified")
    elif save_path is not None:
        #Create a new directory for the models and logs
        model_path = save_path + timestamp + "/models/"
        os.makedirs(save_path + timestamp + "/models/")
        log_path = save_path + timestamp + "/logs_" + timestamp + ".txt"
    elif load_path is not None:
        model_path = load_path + "models/"
        log_path = load_path + "logs_" + timestamp + ".txt"

    logger = Logger(log_path,
        time(),DATASET_ID,"full_dataset",
        model_creator.__name__,
        model_path,
        epochs=epochs,
        lossfunction=lossfunction)

    logger.log("Training samples per class: " + str(train_samples_per_class))
    logger.log("Validation samples per class: " + str(validation_samples_per_class))
    logger.log("Test samples per class: " + str(test_samples_per_class))
    logger.log("Number of experiments: " + str(n_experiments))
    results_all_runs = []
    accuracies = []
    confusion_matrices = []
               
    for n_experiment in range(n_experiments):
        logger.log("Run: " + str(n_experiment))
        columns = ['trial', 'seed', 'comparison1', 'comparison2', 'train_acc', 'val_acc', 'standard', 'prediction1', 'prediction2', 'accuracy']
        logger.log(columns)

        train_pairs = [list((x,y)) for (x,y) in itertools.product(CLASSES, CLASSES) if x < y]
        train_indices = [list((x,y)) for (x,y) in itertools.product([i for i in range(len(CLASSES))], [i for i in range(len(CLASSES))]) if x < y]
        results = pd.DataFrame(columns=columns)

        for n_trial in range(len(train_pairs)):
            seed = (n_trial+1)*(n_experiment+1)
            ########## Create the training pairs ##########
            pair = train_pairs[n_trial]
            index_pair = train_indices[n_trial]
            X_base, y_base = manager.get_raw_with_classes(pair, train_samples_per_class+validation_samples_per_class, shuffle = True, seed=seed)
            y_base = y_base[:,[j for j in range(len(CLASSES)) if j in index_pair]] 
            input_shape = X_base.shape[1:]
            num_classes = y_base.shape[1]
            ########## Train the classifier ##########                
            model = model_creator(input_shape, num_classes)
            if load_path is not None:
                model = load_model(model_path + str(pair) + str(n_experiment) + ".h5")
                # train_accuracy = model.evaluate(X_base, y_base, batch_size = 4)[1]
                # val_accuracy = model.evaluate(X_base, y_base, batch_size = 4)[1]
                # Set train_accuracy and val_accuracy to NAN
                train_accuracy = np.nan
                val_accuracy = np.nan
            else:
            # model.summary()
            # stop_callback = EarlyStopping(monitor = "val_accuracy", patience = epochs, mode = "max", restore_best_weights = True)
                model.compile(loss = lossfunction, optimizer = "rmsprop", metrics = ["accuracy"])
                history = model.fit(
                    X_base, y_base, epochs = epochs, batch_size = 4,
                    validation_split = validation_samples_per_class/(train_samples_per_class+validation_samples_per_class),
                    # callbacks = [stop_callback]
                )

                val_accuracy = history.history['val_accuracy'][-1]
                train_accuracy = history.history['accuracy'][-1]

                try:
                    model.save(model_path + str(pair) + str(n_experiment) + ".h5")
                except:
                    print("could not write model")

            ########## Predict on the remaining classes ##########
            predict_on = [c for c in CLASSES if c not in pair]
            for obj in predict_on:
                X_comp, _ = manager.get_raw_with_classes([obj], test_samples_per_class, shuffle = True, seed=seed)
                dist = model.predict(X_comp)
                choice = np.argmax(dist, axis = 1)
                prediction = np.zeros((choice.size, len(pair)))
                prediction[np.arange(choice.size), choice] = 1
                row = prediction.sum(axis = 0)
                if np.argmax(row) == 0:
                    final_prediction = np.array([1.0,0.0])
                else:
                    final_prediction = np.array([0.0,1.0])                    

                ########## Log the results for each trial ##########
                result = {}
                result["trial"] = n_trial
                result["seed"] = seed
                result["comparison1"] = pair[0]
                result["comparison2"] = pair[1]
                result["train_acc"] = train_accuracy
                result["val_acc"] = val_accuracy
                result["standard"] = obj
                result["prediction1"] = final_prediction[0]
                result["prediction2"] = final_prediction[1]
                result["accuracy"] = (1-abs(compare_hardness(pair[0], pair[1], obj)-final_prediction)[0])*100
                logger.log(", ".join(f"{value:.2f}" if isinstance(value, float) else str(value) for value in result.values()))
                results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

        ########## Log the results for each experiment run ##########
        logger.log(results.to_dict())
        logger.log("Average accuracy:" + str(results["accuracy"].mean()))
        results_all_runs.append(results)
        accuracies.append(results["accuracy"].mean())
        [confusion_matrix, true_confusion_matrix] = compute_confusion_matrix(results)
        confusion_matrices.append(confusion_matrix)
        logger.log("Confusion Matrix" + str(confusion_matrix))
        logger.log("True Confusion Matrix" + str(true_confusion_matrix))
        logger.log("=====================================")

    ########## Log the overall results for all experiment runs ##########
    logger.log("Training samples per class: " + str(train_samples_per_class))
    logger.log("Validation samples per class: " + str(validation_samples_per_class))
    logger.log("Test samples per class: " + str(test_samples_per_class))
    logger.log("Number of experiments: " + str(n_experiments))
    logger.log("Accuracies over all runs:" + str(accuracies))
    logger.log("Average accuracy over all runs:" + str(np.mean(accuracies)))
    logger.log("Standard deviation over all runs:" + str(np.std(accuracies)))
    logger.log("Average Confusion Matrix \n" + str(np.mean(confusion_matrices, axis = 0)))
    return results_all_runs, accuracies, confusion_matrices
        

def run_with_less_data(model_creator, manager, save_path, samples_per_class = 10):
    lossfunction = "categorical_crossentropy"
    model_save_path = save_path + datetime.datetime.fromtimestamp(time()).strftime('%d-%m-%Y_%H-%M-%S') + ".h5"
    logger = Logger(join(LOG_PATH, "less_data.txt"),
    time(),DATASET_ID,"less_data",
    model_creator.__name__,
    model_save_path,
    lossfunction=lossfunction)

    X,y = manager.get_raw()
    input_shape = X.shape[1:]
    num_classes = y.shape[1]
    model = model_creator(input_shape, num_classes)

    train_size = (samples_per_class * len(CLASSES)) / X.shape[0] 
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= train_size, random_state = 42, stratify=y)
    logger.log("Number of Datapoints per class" + str(y_train.sum(axis = 0)))

    stop_callback = EarlyStopping(monitor = "val_accuracy", patience = 100, mode = "max", restore_best_weights = True)

    model.compile(loss = lossfunction, optimizer = "rmsprop", metrics = ["accuracy"])

    history = model.fit(
        X_train, y_train, epochs = 100, batch_size = 4,
        validation_split = 0.1,
        callbacks = [stop_callback]
    )

    try:
        model.save(model_save_path)
    except:
        print("could not write")

    results = model.evaluate(X_test, y_test, batch_size= 4)
    logger.log("test_loss, test_acc = " + str(results))
    logger.log("Required epochs:" + str(len(history.history['loss'])))\
    
    y_hat = model.predict(X)
    matrix = confusion_matrix(predictions = np.argmax(y_hat, axis = 1),labels = np.argmax(y, axis = 1))
    logger.log("Confusion Matrix on less data dataset")
    logger.log(matrix)






if __name__ == "__main__":
    manager = DatasetManager(DATASET_ID, ["C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"], CLASSES)
    # run_with_less_data(get_ConvLSTM, manager, "model_dumps/less_data/", samples_per_class=1)
    
    # run_on_full_dataset(get_ConvLSTM, manager, "model_dumps/without_corrupted_full_dataset/")
    n_experiments = 10
    epochs = 100
    train_samples_per_class = 40
    validation_samples_per_class = 0
    test_samples_per_class = 40

    [results, accuracies, confusion_matrices] = run_train_2_classify_three(
                                                model_creator=get_ConvLSTM, 
                                                manager=manager,
                                                n_experiments=n_experiments,
                                                epochs=epochs,
                                                train_samples_per_class=train_samples_per_class,
                                                validation_samples_per_class=validation_samples_per_class,
                                                test_samples_per_class=test_samples_per_class,
                                                save_path="model_dumps/train_2_classify_three/")