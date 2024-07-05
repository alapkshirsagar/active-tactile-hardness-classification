from models import *
from preprocessing import *
from adaptive_sampling import *

svm = OpticalFlowSVM()
model = svm.get_model()
manager = DatasetManager(["../../IP-Object-Hardness/data_no_noise_vel_02to07_force_1to5/data"], classes=CLASSES, seq_length=4)

X_train, y_train = manager.get_raw_with_classes(CLASSES, 100, shuffle=True, seed=42, augment_factor=1)
X_test, y_test = manager.get_raw_with_classes(CLASSES, 100, shuffle=True, seed=64, augment_factor=1)
