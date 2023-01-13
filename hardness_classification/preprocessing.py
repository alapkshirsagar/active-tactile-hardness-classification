import os
from sklearn.preprocessing import LabelBinarizer
from os.path import join, exists
import numpy as np
import json
import matplotlib.pyplot as plt

SEQ_LENGTH = 5
VISUALIZE_FRAME_SELECTION = True

class DatasetManager():
    def __init__(self, id, path, classes):
        self.id = id
        self.path = path
        self.storage = "processed_data/"
        self.encoder = LabelBinarizer()
        self.encoder.fit(classes)


    
    def __select_frames(self, frames, k, times = [], forces = [], id = 0):

        frame_num = frames.shape[0]

        neutral = frames[int(frame_num * .05)] # neural deformation before pressing

        dis = lambda x: np.linalg.norm( x- neutral)
        distances = np.array(list(map(dis,frames)))


        epsilon = 1400

        first_deformation_index = np.argmax(distances > epsilon)
        max_deformation_index = np.argmax(distances)

        time_span = max_deformation_index - first_deformation_index
        indizes = [int(np.round(first_deformation_index + i *time_span/(k-1))) for i in range(k)]

        if VISUALIZE_FRAME_SELECTION:
            selected_frame_timestamps = times[indizes]
            plt.plot(forces[:, 0], forces[:, 1])
            plt.vlines(selected_frame_timestamps[1:k-1], np.min(forces[:, 1]), np.max(forces[:, 1]), "grey")
            plt.vlines(times[first_deformation_index], np.min(forces[:, 1]), np.max(forces[:, 1]), "green", label="First detected deformation")
            plt.vlines(times[max_deformation_index], np.min(forces[:, 1]), np.max(forces[:, 1]), "red", label="Maximum detected deformation")
            plt.xlabel("Time")
            plt.ylabel("Force on end-effector")
            plt.legend()
            plt.savefig(join(id))
            plt.cla()
            # input("Press Enter to continue...")


        return frames[indizes]


    
    def __get_frames_and_labels(self, path):
        X = []
        Y = []
        meta_path = join(path, "meta/")
        frames_path = join(path, "frames/")
        forces_path = join(path, "forces/")

        for file in os.listdir(meta_path):
            meta = None
            with open(join(meta_path, file),"r") as f:
                meta = json.load(f)

            Y.append(meta["object"])

            frames = np.load(join(frames_path, meta["id"] + "_frames.npy"))
            times = np.load(join(frames_path, meta["id"] + "_times.npy"))
            forces = np.load(join(forces_path, meta["id"] + ".npy"))
            X.append(self.__select_frames(frames, SEQ_LENGTH, times, forces, meta["id"]))
        

        X = np.array(X)
        y = self.encoder.transform(Y) # One hot encoding is in alphabetical order of string labels


        return X,y 


    def get_raw(self):
        if exists(join(self.storage, self.id, "X_raw.npy")):
            X = np.load(join(self.storage, self.id, "X_raw.npy"))
            y = np.load(join(self.storage, self.id, "y_raw.npy"))

            return X,y
        
        X,y = self.__get_frames_and_labels(self.path)

        if not exists(join(self.storage, self.id)):
            os.mkdir(join(self.storage, self.id))
        
        with open(join(self.storage, self.id, "X_raw.npy"), "wb") as f:
            np.save(f,X)

        with open(join(self.storage, self.id, "y_raw.npy"), "wb") as f:
            np.save(f,y)

        return X,y

    def get_raw_with_classes(self, classes):
        X,y = self.get_raw()
        labels = self.encoder.transform(classes)


        X_out = []
        y_out = []

        for i in range(y.shape[0]):  
          if (labels == y[i,:]).all(axis = 1).any(axis = 0): # Weird way of checking if current label (y[i]) is in the list of selected labels (labels)
                X_out.append(X[i])
                y_out.append(y[i])


        return np.array(X_out), np.array(y_out)


        
        
if __name__ == "__main__":
    manager = DatasetManager("43", "data/",["hardness" + str(i+1) for i in range(5)])
    X,y = manager.get_raw_with_classes(["hardness1", "hardness3", "hardness4"])
    print(y[0:20])
    print(X.shape)
    print(y.shape)
