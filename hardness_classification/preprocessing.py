import os
from sklearn.preprocessing import LabelBinarizer
from os.path import join, exists
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate
import imgaug.augmenters as iaa

VISUALIZE_FRAME_SELECTION = False
SELECT_ON_VISUAL_CUES = True
EXCLUDE_CORRUPTED_DATA = True

class DatasetManager():
    def __init__(self, paths, classes, seq_length = None):
        self.id = "Own"
        self.classes = classes
        self.seq_length = seq_length
        self.paths = paths
        self.storage = "./processed_data/"
        # self.storage = "/home/alap/Repositories/2022-07-HapticExploration/ip-hardness/hardness_classification/processed_data/"
        self.encoder = LabelBinarizer()
        self.encoder.fit(classes)


    def __select_frames(self, frames, k, times = [], forces = [], id = 0):

        frame_num = frames.shape[0]

        neutral = frames[int(frame_num * .05)] # neutral deformation before pressing

        dis = lambda x: np.linalg.norm( x- neutral)
        distances = np.array(list(map(dis,frames)))


        epsilon = 1400

        first_deformation_index = np.argmax(distances > epsilon)
        max_deformation_index = np.argmax(distances)
        max_force_timestamp = forces[np.argmin(forces[:, 1]), 0]
        max_force_index = np.argmin(np.absolute(times - max_force_timestamp))
        time_span = max_deformation_index - first_deformation_index
        if k == None:
            if time_span < 3:
                print("Not enough deformation detected for " + str(id) + ". Skipping this data point.")
                return None
            else:
                k = time_span
        if time_span < k:
            print("Not enough deformation detected for " + str(id) + ". Skipping this data point.")
            return None
        if SELECT_ON_VISUAL_CUES:
            indizes = [int(np.round(first_deformation_index + i *time_span/(k-1))) for i in range(k)]
        else:
            indizes = [max_force_index - i for i in range(k)]

        if VISUALIZE_FRAME_SELECTION:
            selected_frame_timestamps = times[indizes]
            plt.plot(forces[:, 0], forces[:, 1])
            
            if SELECT_ON_VISUAL_CUES:
                plt.vlines(times[first_deformation_index], np.min(forces[:, 1]), np.max(forces[:, 1]), "green", label="First detected deformation")
                plt.vlines(times[max_deformation_index], np.min(forces[:, 1]), np.max(forces[:, 1]), "red", label="Maximum detected deformation")
                
            plt.vlines(selected_frame_timestamps, np.min(forces[:, 1]), np.max(forces[:, 1]), "grey", label="Selected frames")
            plt.vlines(max_force_timestamp, np.min(forces[:, 1]), np.max(forces[:, 1]), "red", label="Maximum force timestamp")
            plt.xlabel("Time")
            plt.ylabel("Force on end-effector")
            plt.legend()
            # plt.show()
            plt.savefig(join(id))
            plt.cla()
            # input("Press Enter to continue...")

        return frames[indizes]


    def __check_validity(self, forces):
        timestamps = forces[:, 0]

        time_between_timestamps = timestamps[1:] - timestamps[:-1]

        return np.max(time_between_timestamps) < 100 # milliseconds

    
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
            forces = np.load(join(forces_path, meta["id"] + ".npy"))

            if (not EXCLUDE_CORRUPTED_DATA) or self.__check_validity(forces):
                frames = np.load(join(frames_path, meta["id"] + "_frames.npy"))
                times = np.load(join(frames_path, meta["id"] + "_times.npy"))
                selected_frames = self.__select_frames(frames, self.seq_length, times, forces, meta["id"])
                if selected_frames is not None:
                    # Check if selected_frames have the correct shape
                    assert selected_frames.shape[1] == 240
                    assert selected_frames.shape[2] == 320
                    assert selected_frames.shape[3] == 3
                    X.append(selected_frames)
                    Y.append(meta["object"])   
            else:
                print("Data point " + str(meta["id"]) + " looks corrupted and is excluded.")

        X = np.array(X)
        y = np.array(Y)

        return X,y 


    def get_raw(self):
        if exists(join(self.storage, self.id, "X_"+str(self.seq_length)+"_raw.npy")):
            X = np.load(join(self.storage, self.id, "X_"+str(self.seq_length)+"_raw.npy"), allow_pickle=True)
            y = np.load(join(self.storage, self.id, "y_"+str(self.seq_length)+"_raw.npy"), allow_pickle=True)

            return X,y

        Xs = []
        ys = []
        for path in self.paths:
            f_and_l = self.__get_frames_and_labels(path)
            Xs.append(f_and_l[0])
            ys.append(f_and_l[1])
        
        X = []
        y = []
        for index in range(len(Xs)):
            for i in range(Xs[index].shape[0]):
                X.append(Xs[index][i])
                y.append(ys[index][i])

        X = np.array(X)
        y = np.array(y)

        if not exists(join(self.storage, self.id)):
            os.mkdir(join(self.storage, self.id))
        
        with open(join(self.storage, self.id, "X_"+str(self.seq_length)+"_raw.npy"), "wb") as f:
            np.save(f,X)

        with open(join(self.storage, self.id, "y_"+str(self.seq_length)+"_raw.npy"), "wb") as f:
            np.save(f,y)

        return X,y


    def get_raw_with_classes(self, classes, num_samples_per_class, shuffle=False, seed=None, augment_factor=0):
        X,y = self.get_raw()
        labels = classes
        # Create a list of indices and shuffle it if necessary
        indices = list(range(y.shape[0]))
        if shuffle:
            random.seed(seed)
            random.shuffle(indices)


        X_out = []
        y_out = []
        # Initialize counters for each class
        class_counts = {tuple(label): 0 for label in labels}

        for i in indices:
            # Get the current label
            current_label = y[i]
            # Check if current label is in the list of selected labels and the class count is less than the specified number
            if current_label in labels:
                if class_counts[tuple(current_label)] < num_samples_per_class:
                    X_out.append(X[i])
                    y_out.append(y[i])
                    # Increment the count for the current class
                    class_counts[tuple(current_label)] += 1
            # Break the loop if all classes have enough samples
            if all(count >= num_samples_per_class for count in class_counts.values()):
                break
        if augment_factor:
            X_out, y_out = self.augment_data(X_out, y_out, augment_factor=augment_factor)
        return np.array(X_out), np.array(y_out)

    def augment_data(self, X, y, augment_factor):
        # X is assumed to be a list of 4D numpy arrays (num_images, height, width, channels)
        # y is assumed to be a list of 1D numpy arrays (num_images, num_classes)
        # Initialize output lists
        X_aug = []
        y_aug = []

        # Iterate over each sample in the list
        for s in range(len(X)):
            # Add 10 rotations of each image in the sample
            for angle in range(0, 360, int(360/augment_factor)): # This will give 10 equally spaced angles between 0 and 360
                rotated_sample = np.zeros((X[s].shape[0], X[s].shape[1], X[s].shape[2], X[s].shape[3]))
                for i in range(X[s].shape[0]):
                    rotated_img = rotate(X[s][i], angle, reshape=False) # Rotate the image without changing its shape
                    rotated_sample[i] = rotated_img
                X_aug.append(rotated_sample)
                y_aug.append(y[s])
        return X_aug, y_aug

if __name__ == "__main__":
    manager = DatasetManager("no_noise__visual_selection__filtered", ["data/data_no_noise/"], ["hardness" + str(i+1) for i in range(5)])
    X,y = manager.get_raw_with_classes(["hardness1", "hardness2", "hardness3", "hardness4", "hardness5"], 100)
    # print(y[0:20])
    print("Shape of X: " + str(X.shape))
    print("Shape of y: " + str(y.shape))
