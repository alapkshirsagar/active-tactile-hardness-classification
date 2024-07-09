import os
from sklearn.preprocessing import LabelBinarizer
from os.path import join, exists
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate
import cv2
from utils import compare_hardness_pair, compare_hardness_triplet
from sklearn.model_selection import train_test_split

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
        self.objects = "Full Dataset"


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
    
    def get_ordered_pairs(self, classes, num_samples_per_class, shuffle=False, seed=None, augment_factor=0):
        """
        Given a list of classes and the number of samples per class, this function returns a list of pairs of samples and their labels. The label is the index of the harder sample in the pair.
        inputs:
        classes: list of classes to be considered
        num_samples_per_class: number of samples per class
        shuffle: whether to shuffle the samples
        seed: seed for the random number generator
        augment_factor: factor to augment the data
        returns:
        X_pairs: list of pairs of samples
        y_pairs: list of labels for the pairs
        """
        X, y = self.get_raw_with_classes(classes, num_samples_per_class, shuffle, seed, augment_factor)
        X_pairs = []
        y_pairs = []
        for i in range(len(y)):
            for j in range(i+1, len(y)):
                X_pairs.append([X[i], X[j]])
                if compare_hardness_pair(str(y[i]), str(y[j])) == 0:
                    y_pairs.append(0)
                else:
                    y_pairs.append(1)
        return np.array(X_pairs), np.array(y_pairs)
    
    def get_ordered_triplets(self, classes, num_samples_per_class, shuffle=False, seed=None, augment_factor=0):
        """
        Given a list of classes and the number of samples per class, this function returns a list of triplets of samples and their labels. Each triplet is ordered (comparison1, comparison2, standard). The label is the index of the comparison that is closer to the standard.
        inputs:
        classes: list of classes to be considered
        num_samples_per_class: number of samples per class
        shuffle: whether to shuffle the samples
        seed: seed for the random number generator
        augment_factor: factor to augment the data
        returns:
        X_triplets: list of triplets of samples
        y_triplets: list of labels for the triplets
        """
        X, y = self.get_raw_with_classes(classes, num_samples_per_class, shuffle, seed, augment_factor)
        X_triplets = []
        y_triplets = []
        for i in range(len(y)):
            for j in range(i+1, len(y)):
                for k in range(j+1, len(y)):
                    X_triplets.append([X[i], X[j], X[k]])
                    if compare_hardness_triplet(str(y[i]), str(y[j]), str(y[k]), dataset="Own") == 0:
                        y_triplets.append(0)
                    else:
                        y_triplets.append(1)
        return np.array(X_triplets), np.array(y_triplets)

    

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

class DatasetManagerYuan():
    def __init__(self, path, objects = [], classes = [8, 11, 17, 27, 32, 46, 48, 50, 62, 73, 81, 87, 95], seq_length = 5):
        self.id = "Yuan"
        self.path = path
        self.storage = "/home/alap/Repositories/2022-07-HapticExploration/ip-hardness/hardness_classification/processed_data/"
        self.encoder = LabelBinarizer()
        self.objects = objects
        self.encoder.fit(classes)
        self.classes = classes
        self.seq_length = seq_length

    def __select_frames(self, frames, k, file):
        init_frame_num = 0
        neutral = np.linalg.norm(frames[int(init_frame_num)])
        dis = lambda x: np.linalg.norm( np.linalg.norm(x)- neutral)
        distances = np.array(list(map(dis,frames)))
        epsilon = 600
        first_deformation_index = np.argmax(distances > epsilon)
        max_deformation_index = np.argmax(distances)
        print(file)
        time_span = max_deformation_index - first_deformation_index
        if k == None:
            if time_span < 3:
                print("Not enough deformation detected for " + str(file) + ". Skipping this data point.")
                return None
            else:
                k = time_span
        if time_span < k:
            print("Not enough deformation detected for " + str(file) + ". Skipping this data point.")
            return None
        indizes = [int(np.round(first_deformation_index + i *time_span/(k-1))) for i in range(k)]
        return frames[indizes]
    
    def __read_frames_from_avi(self, file_path):
        # Open the video file
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        frames = []

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If the frame was not read correctly, break the loop
            if not ret:
                break
            # Downsample the frame to 240x320
            frame = cv2.resize(frame, (320, 240))

            # Append the frame to the list
            frames.append(frame)

        # Release the video capture object
        cap.release()

        # Convert the list of frames to a NumPy array
        frames_array = np.array(frames)

        return frames_array
    
    def __get_frames_and_labels(self, path, object):
        X = []
        Y = []
        for file in os.listdir(path):
            # If the first segment of the filename before "_" matches the object name
            if file.split("_")[0] == object:
                # Load the frames from .avi file
                frames = self.__read_frames_from_avi(join(path, file))
                selected_frames = self.__select_frames(frames, self.seq_length, file)
                if selected_frames is not None:
                    Y.append(int(file.split("_")[1]))
                    X.append(selected_frames)
        X = np.array(X)
        y = Y
        return X,y


    def get_raw(self):
        X = []
        y = []
        for obj in self.objects:
            if exists(join(self.storage, self.id, obj, "X_"+str(self.seq_length)+"_raw.npy")):
                X_obj = np.load(join(self.storage, self.id, obj, "X_"+str(self.seq_length)+"_raw.npy"), allow_pickle=True)
                y_obj = np.load(join(self.storage, self.id, obj, "y_"+str(self.seq_length)+"_raw.npy"), allow_pickle=True)
            else:
                f_and_l = self.__get_frames_and_labels(self.path, obj)
                X_obj = np.array(f_and_l[0])
                y_obj = np.array(f_and_l[1])
                if not exists(join(self.storage, self.id)):
                    os.mkdir(join(self.storage, self.id))
                if not exists(join(self.storage, self.id, obj)):
                    os.mkdir(join(self.storage, self.id, obj))
                with open(join(self.storage, self.id, obj, "X_"+str(self.seq_length)+"_raw.npy"), "wb") as f:
                    np.save(f,X_obj)
                with open(join(self.storage, self.id, obj, "y_"+str(self.seq_length)+"_raw.npy"), "wb") as f:
                    np.save(f,y_obj)
            X.append(X_obj)
            y.append(y_obj)
        X = np.concatenate(X)
        y = np.concatenate(y)
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
        class_counts = {label: 0 for label in labels}

        for i in indices:
            # Get the current label
            current_label = y[i]
            # Check if current label is in the list of selected labels and the class count is less than the specified number
            if (labels == current_label).any(axis=0):
                if class_counts[current_label] < num_samples_per_class:
                    X_out.append(X[i])
                    y_out.append(y[i])
                    # Increment the count for the current class
                    class_counts[current_label] += 1
            # Break the loop if all classes have enough samples
            if all(count >= num_samples_per_class for count in class_counts.values()):
                break
        # Check if number of samples per class is reached, if not raise an error
        if not all(count >= num_samples_per_class for count in class_counts.values()):
            raise ValueError("Not enough samples for each class. Please reduce the number of samples per class.")

        if augment_factor:
            X_out, y_out = self.augment_data(X_out, y_out, augment_factor=augment_factor)

        return np.array(X_out), np.array(y_out)

    def get_ordered_pairs(self, classes, num_train_samples_per_class, num_val_samples_per_class, shuffle=False, seed=None, augment_factor=0):
        """
        Given a list of classes and the number of samples per class, this function returns a list of pairs of samples and their labels. The label is the index of the harder sample in the pair.
        inputs:
        classes: list of classes to be considered
        num_train_samples_per_class: number of training samples per class
        num_val_samples_per_class: number of validation samples per class
        shuffle: whether to shuffle the samples
        seed: seed for the random number generator
        augment_factor: factor to augment the data
        returns:
        X_pairs: list of pairs of samples
        y_pairs: list of labels for the pairs
        """
        X, y = self.get_raw_with_classes(classes, num_train_samples_per_class+num_val_samples_per_class, shuffle, seed, augment_factor)
        # Split the data into training and validation sets, make sure to stratify the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=num_train_samples_per_class/(num_train_samples_per_class+num_val_samples_per_class), stratify=y, random_state=seed)

        X_train_pairs = []
        y_train_pairs = []
        for i in range(len(y_train)):
            for j in range(i+1, len(y_train)):
                X_train_pairs.append([X_train[i], X_train[j]])
                if compare_hardness_pair(str(y_train[i]), str(y_train[j]), dataset="Yuan") == 0:
                    y_train_pairs.append(0)
                else:
                    y_train_pairs.append(1)
        X_val_pairs = []
        y_val_pairs = []
        for i in range(len(y_val)):
            for j in range(i+1, len(y_val)):
                X_val_pairs.append([X_val[i], X_val[j]])
                if compare_hardness_pair(str(y_val[i]), str(y_val[j]), dataset="Yuan") == 0:
                    y_val_pairs.append(0)
                else:
                    y_val_pairs.append(1)
        return np.array(X_train_pairs), np.array(y_train_pairs), np.array(X_val_pairs), np.array(y_val_pairs)
    
    def get_ordered_triplets(self, classes, num_train_samples_per_class, num_val_samples_per_class, shuffle=False, seed=None, augment_factor=0):
        """
        Given a list of classes and the number of samples per class, this function returns a list of triplets of samples and their labels. Each triplet is ordered (comparison1, comparison2, standard). The label is the index of the comparison that is closer to the standard.
        inputs:
        classes: list of classes to be considered
        num_train_samples_per_class: number of training samples per class
        num_val_samples_per_class: number of validation samples per class
        shuffle: whether to shuffle the samples
        seed: seed for the random number generator
        augment_factor: factor to augment the data
        returns:
        X_triplets: list of triplets of samples
        y_triplets: list of labels for the triplets
        """
        X, y = self.get_raw_with_classes(classes, num_train_samples_per_class+num_val_samples_per_class, shuffle, seed, augment_factor)
        # Split the data into training and validation sets, make sure to stratify the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=num_train_samples_per_class/(num_train_samples_per_class+num_val_samples_per_class), stratify=y, random_state=seed)

        X_train_triplets = []
        y_train_triplets = []
        for i in range(len(y_train)):
            for j in range(i+1, len(y_train)):
                for k in range(j+1, len(y_train)):
                    X_train_triplets.append([X_train[i], X_train[j], X_train[k]])
                    if compare_hardness_triplet(str(y_train[i]), str(y_train[j]), str(y_train[k]), dataset="Yuan") == 0:
                        y_train_triplets.append(0)
                    else:
                        y_train_triplets.append(1)
        X_val_triplets = []
        y_val_triplets = []
        for i in range(len(y_val)):
            for j in range(i+1, len(y_val)):
                for k in range(j+1, len(y_val)):
                    X_val_triplets.append([X_val[i], X_val[j], X_val[k]])
                    if compare_hardness_triplet(str(y_val[i]), str(y_val[j]), str(y_val[k]), dataset="Yuan") == 0:
                        y_val_triplets.append(0)
                    else:
                        y_val_triplets.append(1)

        return np.array(X_train_triplets), np.array(y_train_triplets), np.array(X_val_triplets), np.array(y_val_triplets)






if __name__ == "__main__":
    # manager = DatasetManager("no_noise__visual_selection__filtered", ["data/data_no_noise/"], ["hardness" + str(i+1) for i in range(5)])
    # X,y = manager.get_raw_with_classes(["hardness1", "hardness2", "hardness3", "hardness4", "hardness5"])
    classes = [8, 11, 17, 19, 27, 32, 36, 46, 48, 50, 54, 62, 65, 71, 73, 74, 81, 87, 95] # These are the classes in the Yuan dataset
    # classes = [8, 27, 48, 87, 95] # Use these for FLAT object
    # classes = [8, 27, 46, 71, 87] # Use these for CHOCO object
    classes = [8, 11, 17, 27, 46, 48, 62, 73, 81, 87, 95] # Use these for FLAT object, 11 classes

    manager = DatasetManagerYuan("/home/alap/Repositories/2022-07-HapticExploration/dataset-yuan/Data1/", objects = ["FLAT"], classes = classes, seq_length=4)
    X,y = manager.get_raw_with_classes(classes=classes, num_samples_per_class=20, shuffle=True, seed=42, augment_factor=0)
    # classes = ["hardness" + str(i+1) for i in range(5)]
    # manager = DatasetManager(["/home/alap/Repositories/2022-07-HapticExploration/ip-hardness/data_collection/data/IP-Object-Hardness/data_no_noise_vel_02to07_force_1to5/data/"], classes = ["hardness" + str(i+1) for i in range(5)], seq_length=None)
    # X,y = manager.get_raw_with_classes(classes=classes, num_samples_per_class=40, shuffle=True, seed=42, augment_factor=0)
    # print(y[0:20])
    print("Shape of X: " + str(X.shape))
    print("Shape of y: " + str(y.shape))
    # Print the number of samples for each class, classes use string labels
    for label in classes:
        print(str(label) + ": " + str(np.sum(y == label)))

