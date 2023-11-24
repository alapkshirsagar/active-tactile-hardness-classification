from preprocessing import *
from models import *

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.math import confusion_matrix
import numpy as np
import itertools
from time import time
import datetime
from log import Logger
import uuid


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

def run_train_2_classify_three(model_creator, manager, save_path):
    train_pairs = [list((x,y)) for (x,y) in itertools.product(CLASSES, CLASSES) if x < y]
    train_indizes = [list((x,y)) for (x,y) in itertools.product([i for i in range(len(CLASSES))], [i for i in range(len(CLASSES))]) if x < y]
    
    f = open("logs/train_2_classify_three.txt", "a")
    f.write("=============================================\n")
    f.write("Time: " + str(time()) + "\n")
    f.write("Exp: train_2_classify_three\n")
    f.write("Dataset ID: " + str(DATASET_ID) + "\n")
    f.close()

    
    print(train_indizes)

    for i in range(len(train_pairs)):
        pair = train_pairs[i]
        predict_on = [c for c in CLASSES if c not in pair]
        index_pair = train_indizes[i]

        X_base, y_base = manager.get_raw_with_classes(pair)
        y_base = y_base[:,[j for j in range(len(CLASSES)) if j in index_pair]] 

        print(predict_on)
        predict_on_data = []
        for obj in predict_on:
            print(obj)
            X_comp, _ = manager.get_raw_with_classes([obj])
            predict_on_data.append((obj, X_comp))


       

        input_shape = X_base.shape[1:]
        num_classes = y_base.shape[1]

        model = model_creator(input_shape, num_classes)
        model.summary()
        X_train, X_test, y_train, y_test = train_test_split(X_base,y_base, train_size = .8)


        stop_callback = EarlyStopping(monitor = "val_accuracy", patience = 100, mode = "max", restore_best_weights = True)

        model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
        history = model.fit(
            X_train, y_train, epochs = 100, batch_size = 4,
            validation_split = 0.1,
            callbacks = [stop_callback]
        )
        try:
            model.save(save_path + str(pair) + ".h5")
        except:
            print("could not write")

        loss, acc = model.evaluate(X_test, y_test, batch_size= 4)

        f = open("logs/train_2_classify_three.txt", "a")

        print("================ Trained on ", str(pair), "====== Predict on ", str(predict_on),"========\n")
        f.write("==> Trained on " + str(pair) +  " predict on " + str(predict_on) + "\n")
        print("====== test_loss ", loss)
        f.write("--> test_loss " + str(loss) + "\n")
        print("====== test_acc ", acc)
        f.write("--> test_acc " + str(acc) + "\n")

        for obj, data in predict_on_data:
            dist = model.predict(data)
            choice = np.argmax(dist, axis = 1)
            prediction = np.zeros((choice.size, len(pair)))
            prediction[np.arange(choice.size), choice] = 1
            row = prediction.sum(axis = 0)

            print("======== Prediction for ", obj, ":", row)
            f.write("--> Prediction for " + obj + " : " + str(row) + "\n")

        f.close()
        

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
    # print("Here we go")
    manager = DatasetManager(DATASET_ID, ["/home/alap/Repositories/2022-07-HapticExploration/ip-hardness/data_collection/data/IP-Object-Hardness/data_no_noise_vel_02to07_force_1to5/data/"], CLASSES)
    run_with_less_data(get_ConvLSTM, manager, "model_dumps/less_data/", samples_per_class=1)
    
    # run_on_full_dataset(get_ConvLSTM, manager, "model_dumps/without_corrupted_full_dataset/")
