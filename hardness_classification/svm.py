import torch
from sklearn.metrics import confusion_matrix
from preprocessing import *
from models import *
from adaptive_sampling import *
from time import time
import wandb

manager = DatasetManager(["C:\\MLData\\data_no_noise_vel_02to07_force_1to5\\data"], classes=CLASSES, seq_length=4)
def train_and_test_on_all_cls():
    svm = OpticalFlowSVM()
    X_train, y_train = manager.get_raw_with_classes(CLASSES, 80, seed=time())
    y_train = manager.encoder.transform(y_train)

    X_test, y_test = manager.get_raw_with_classes(CLASSES, 200, seed=time())
    y_test = manager.encoder.transform(y_test)
    y_test = np.argmax(y_test, axis=1)

    svm.compile_model() #placeholder

    for i in range(1,33):
        svm.train_model(X_train, y_train, i, 10)
        # test the accuracy and print the confusion matrix
        y_pred = svm.predict(X_test)
        print("i: ", i, "Accuracy: ", accuracy_score(y_test, y_pred))
def leave_one_out(comp_cls_id, training_size):
    base_cls=[CLASSES[j] for j in range(5) if j != comp_cls_id]
    test_cls=[CLASSES[comp_cls_id]]

    svm = OpticalFlowSVM()
    svm.get_model()
    X_train, y_train = manager.get_raw_with_classes(CLASSES, 50, seed=time())
    X_base, y_base, _, _ = base_comp_split(X_train, y_train, base_cls,[])
    y_base = manager.encoder.transform(y_base)

    X_test, y_test = manager.get_raw_with_classes(CLASSES, 200, seed=time())

    X_test_b, y_test_b, X_test_c, y_test_c = base_comp_split(X_test, y_test, base_cls, test_cls)

    y_test = manager.encoder.transform(y_test)
    y_test = np.argmax(y_test, axis=1)
    y_test_b = manager.encoder.transform(y_test_b)
    y_test_b = np.argmax(y_test_b, axis=1)
    y_test_c = manager.encoder.transform(y_test_c)
    y_test_c = np.argmax(y_test_c, axis=1)

    svm.train_model(X_base, y_base, training_size, 10)

    y_pred = svm.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    y_pred_b = svm.predict(X_test_b)
    base_acc = accuracy_score(y_test_b, y_pred_b)
    print("acc on base class: ", base_acc)
    y_pred_c = svm.predict(X_test_c)
    comp_acc = compute_acc(y_pred_c, y_test_c,  hardness_levels)
    print("acc on comp class: ", comp_acc)

    return base_acc, comp_acc



if __name__ == "__main__":
    run = wandb.init(project="SVM", name=f"run_{time()}")
    for training_size in range(1,33):
        avg_base_acc = 0
        avg_comp_acc = 0
        for i in range(5):
            base_acc, comp_acc = leave_one_out(i, training_size)
            avg_base_acc += base_acc
            avg_comp_acc += comp_acc
        avg_base_acc /= 5
        avg_comp_acc /= 5
        run.log({"training_size": training_size, "avg_base_acc": avg_base_acc, "avg_comp_acc": avg_comp_acc})
    run.finish()