import numpy as np

dist = np.array([
    [0.2,.4,.4],
    [.4,.2,.2],
    [.2,.2,.4]
])

CLASSES = [1,2,3]

choice = np.argmax(dist, axis = 1)
prediction = np.zeros((choice.size, len(CLASSES)))
prediction[np.arange(choice.size), choice] = 1

print(prediction)