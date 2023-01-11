from preprocessing import *
from models import *


from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


DATASET_ID = "42"
CLASSES = ["hardness" + str(i+1) for i in range(5)]



def run_on_full_dataset(model_creator, manager, save_path):
    X,y = manager.get_raw()
    input_shape = X.shape[1:]
    num_classes = y.shape[1]
    model = model_creator(input_shape, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 42)
    print("Number of Datapoints per class",y_train.sum(axis = 0))

    stop_callback = EarlyStopping(monitor = "val_accuracy", patience = 30, mode = "max", restore_best_weights = True)

    model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
    history = model.fit(
        X_train, y_train, epochs = 100, batch_size = 4,
        validation_split = 0.1,
        callbacks = [stop_callback]
    )

    #model.save(save_path)


    results = model.evaluate(X_test, y_test, batch_size= 4)
    print(results)

    # TODO Print Confusion Matrix

def run_leave_one_out(model_creator, manager, save_path):
    confusion = np.zeros((len(CLASSES), len(CLASSES)))

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

        model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
        history = model.fit(
            X_train, y_train, epochs = 100, batch_size = 4,
            validation_split = 0.1,
            callbacks = [stop_callback]
        )
        try:
            model.save(save_path + "without_" + comp_class + ".h5")
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
    manager = DatasetManager(DATASET_ID, "data/", CLASSES)
    run_leave_one_out(get_ConvLSTM, manager, "model_dumps/leave_one_out/")

