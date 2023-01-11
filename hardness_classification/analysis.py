
from tensorflow.keras.models import load_model

from tensorflow.math import confusion_matrix

from preprocessing import DatasetManager

import matplotlib.pyplot as plt 

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split



manager = DatasetManager("42", "data/", ["hardness" + str(i + 1) for i in range(5)])
model = load_model("ConvLSTM.h5")

X,y = manager.get_raw()

y_hat = model.predict(X)

print(y_hat.shape)
print(y.shape)

matrix = confusion_matrix(predictions = np.argmax(y_hat, axis = 1),labels = np.argmax(y, axis = 1))
print("Confusion Matrix on full dataset")
print(matrix)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 42, stratify= y)
results = model.evaluate(X_test, y_test, batch_size= 4)
print("Test Acc:", results[1])
y_test_hat = model.predict(X_test)
matrix = confusion_matrix(predictions = np.argmax(y_test_hat, axis = 1),labels = np.argmax(y_test, axis = 1))
print("Confusion Matrix on test data")
print(matrix)


