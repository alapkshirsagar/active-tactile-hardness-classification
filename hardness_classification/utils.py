import pandas as pd
import numpy as np
from itertools import combinations, permutations

# True hardness values: 
#Deform level 1 (most soft): 0.44
#Deform level 2: 0.68
#Deform level 3: 0.79
#Deform level 4: 1.02
#Deform level 5 (least soft): 1.13
# Hardness values for 5 objects (own dataset)
# hardness_values_own = {"hardness1":0.44, 
#                    "hardness2":0.68, 
#                    "hardness3":0.79, 
#                    "hardness4":1.02, 
#                    "hardness5":1.13}  # Replace with actual hardness values
hardness_values_own = {"hardness1":1.13, 
                   "hardness2":1.02, 
                   "hardness3":0.79, 
                   "hardness4":0.68, 
                   "hardness5":0.44}  # Replace with actual hardness values

# True hardness values for (Yuan dataset):
hardness_values_yuan = {"8": 8,
                    "11": 11, 
                    "17": 17, 
                    "27": 27,
                    "32": 32, 
                    "46": 46,
                    "48": 48,
                    "50": 50, 
                    "62": 62,
                    "71": 71,
                    "73": 73,
                    "81": 81,
                    "87": 87,
                    "95": 95} 

def generate_comparison_triplets(classes):
    # Generate all combinations of the 5 objects taken 2 at a time for comparison
    comparison_pairs = list(combinations(classes, 2))

    # List to hold the triplets
    comparison_triplets = []

    # Generate triplets
    for comparison_pair in comparison_pairs:
        for standard in classes:
            if standard not in comparison_pair:
                comparison_triplets.append((*comparison_pair, standard))

    print("Shape of comparison_triplets: ", len(comparison_triplets))
    print("Comparison triplets:", comparison_triplets)
    return comparison_triplets

# Function to return similarity results for a triplet
def compare_hardness_triplet(comparison1, comparison2, standard, dataset):
    if dataset == "Own":
        hardness_values = hardness_values_own
    elif dataset == "Yuan":
        hardness_values = hardness_values_yuan
    else:
        raise ValueError("Invalid dataset")
    closer_to_comparison1 = abs(hardness_values[comparison1] - hardness_values[standard]) < abs(hardness_values[comparison2] - hardness_values[standard])
    return 0 if closer_to_comparison1 else 1

# Function to return similarity results for a pair, returns 0 if the comparison1 is harder than comparison2, returns 1 otherwise
def compare_hardness_pair(comparison1, comparison2, dataset):
    if dataset == "Own":
        hardness_values = hardness_values_own
    elif dataset == "Yuan":
        hardness_values = hardness_values_yuan
    else:
        raise ValueError("Invalid dataset")
    return (0 if hardness_values[comparison1] > hardness_values[comparison2] else 1)

# Function to return similarity results for any number of comparison classes, returns the index of the closest comparison class
def compare_hardness(comparison_classes, standard, dataset):
    # Convert comparison_classes and standard to strings
    comparison_classes = [str(class_) for class_ in comparison_classes]
    standard = str(standard)

    if dataset == "Own":
        hardness_values = hardness_values_own
    elif dataset == "Yuan":
        hardness_values = hardness_values_yuan
    else:
        raise ValueError("Invalid dataset")
    closest_comparison = comparison_classes[0]
    min_diff = abs(hardness_values[closest_comparison] - hardness_values[standard])
    for comparison in comparison_classes[1:]:
        diff = abs(hardness_values[comparison] - hardness_values[standard])
        if diff < min_diff:
            min_diff = diff
            closest_comparison = comparison
    # Get the index of the closest comparison class
    closest_comparison_index = comparison_classes.index(closest_comparison)
    return closest_comparison_index

def compute_confusion_matrix_triplet(dataframe, dataset):
    classes = pd.concat([dataframe['comparison1'], dataframe['comparison2']]).unique()
    # Convert classes to strings
    classes = [str(class_) for class_ in classes]
    confusion_matrix = pd.DataFrame(0.0, 
                                    index=classes, 
                                    columns=classes)
    true_confusion_matrix = pd.DataFrame(0.0, 
                                    index=classes, 
                                    columns=classes)
    comparison_triplets = generate_comparison_triplets(classes)
    for triplet in comparison_triplets:
        comparison1_object = triplet[0]
        comparison2_object = triplet[1]
        standard_object = triplet[2]
        true_prediction = compare_hardness_triplet(comparison1_object, comparison2_object, standard_object, dataset)
        if true_prediction == 0:
            true_prediction1 = 1
            true_prediction2 = 0
        else:
            true_prediction1 = 0
            true_prediction2 = 1
        trial_row = dataframe.loc[
            (dataframe['standard'].astype(str) == standard_object) &
            (dataframe['comparison1'].astype(str) == comparison1_object) &
            (dataframe['comparison2'].astype(str) == comparison2_object)
        ]
        confusion_matrix.at[standard_object, comparison1_object] += trial_row['prediction1'].iat[0]
        confusion_matrix.at[standard_object, comparison2_object]+= trial_row['prediction2'].iat[0]

        true_confusion_matrix.at[standard_object, comparison1_object] += true_prediction1*(trial_row['prediction1'].iat[0] + trial_row['prediction2'].iat[0])
        true_confusion_matrix.at[standard_object, comparison2_object] += true_prediction2*(trial_row['prediction1'].iat[0] + trial_row['prediction2'].iat[0])
    # Normalize the rows of the confusion matrix
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)
    true_confusion_matrix = true_confusion_matrix.div(true_confusion_matrix.sum(axis=1), axis=0)
    return confusion_matrix, true_confusion_matrix

def compute_confusion_matrix_leave_one_out(dataframe, dataset):
    classes = pd.concat([dataframe['standard']]).unique()
    # Convert classes to strings
    classes = [str(class_) for class_ in classes]
    confusion_matrix = pd.DataFrame(0.0, 
                                    index=classes, 
                                    columns=classes)
    true_confusion_matrix = pd.DataFrame(0.0, 
                                    index=classes, 
                                    columns=classes)
    for index, row in dataframe.iterrows(): 
        standard_object = str(row['standard'])
        comparison = [str(class_) for class_ in row['comparisons']]
        prediction = row['prediction']

        true_prediction = compare_hardness(comparison, standard_object, dataset)
        confusion_matrix.at[standard_object, comparison[prediction]] += 1.0
        true_confusion_matrix.at[standard_object, comparison[true_prediction]] += 1.0
    # Normalize the rows of the confusion matrix
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)
    true_confusion_matrix = true_confusion_matrix.div(true_confusion_matrix.sum(axis=1), axis=0)
    return confusion_matrix, true_confusion_matrix 

def compute_accuracy(dataframe, dataset):
    accuracy = []
    classes = pd.concat([dataframe['comparison1'], dataframe['comparison2']]).unique()
    classes = [str(class_) for class_ in classes]

    comparison_triplets = generate_comparison_triplets(classes)
    for triplet in comparison_triplets:
        comparison1_object = triplet[0]
        comparison2_object = triplet[1]
        standard_object = triplet[2]
        [true_prediction1, true_prediction2] = compare_hardness_triplet(comparison1_object, comparison2_object, standard_object, dataset)
        trial_row = dataframe.loc[
            (dataframe['standard'].astype(str) == standard_object) &
            (dataframe['comparison1'].astype(str) == comparison1_object) &
            (dataframe['comparison2'].astype(str) == comparison2_object)
        ]
        accuracy.append(1-abs(true_prediction1-trial_row['prediction1']/(trial_row['prediction1']+trial_row['prediction2'])))
        print(accuracy[-1])
    
    
    return np.mean(np.array(accuracy))



if __name__ == "__main__":
    # Your DataFrame
    dataframe = {'trial': {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 4, 13: 4, 14: 4, 15: 5, 16: 5, 17: 5, 18: 6, 19: 6, 20: 6, 21: 7, 22: 7, 23: 7, 24: 8, 25: 8, 26: 8, 27: 9, 28: 9, 29: 9}, 'seed': {0: 9, 1: 9, 2: 9, 3: 18, 4: 18, 5: 18, 6: 27, 7: 27, 8: 27, 9: 36, 10: 36, 11: 36, 12: 45, 13: 45, 14: 45, 15: 54, 16: 54, 17: 54, 18: 63, 19: 63, 20: 63, 21: 72, 22: 72, 23: 72, 24: 81, 25: 81, 26: 81, 27: 90, 28: 90, 29: 90}, 'comparison1': {0: 'hardness1', 1: 'hardness1', 2: 'hardness1', 3: 'hardness1', 4: 'hardness1', 5: 'hardness1', 6: 'hardness1', 7: 'hardness1', 8: 'hardness1', 9: 'hardness1', 10: 'hardness1', 11: 'hardness1', 12: 'hardness2', 13: 'hardness2', 14: 'hardness2', 15: 'hardness2', 16: 'hardness2', 17: 'hardness2', 18: 'hardness2', 19: 'hardness2', 20: 'hardness2', 21: 'hardness3', 22: 'hardness3', 23: 'hardness3', 24: 'hardness3', 25: 'hardness3', 26: 'hardness3', 27: 'hardness4', 28: 'hardness4', 29: 'hardness4'}, 'comparison2': {0: 'hardness2', 1: 'hardness2', 2: 'hardness2', 3: 'hardness3', 4: 'hardness3', 5: 'hardness3', 6: 'hardness4', 7: 'hardness4', 8: 'hardness4', 9: 'hardness5', 10: 'hardness5', 11: 'hardness5', 12: 'hardness3', 13: 'hardness3', 14: 'hardness3', 15: 'hardness4', 16: 'hardness4', 17: 'hardness4', 18: 'hardness5', 19: 'hardness5', 20: 'hardness5', 21: 'hardness4', 22: 'hardness4', 23: 'hardness4', 24: 'hardness5', 25: 'hardness5', 26: 'hardness5', 27: 'hardness5', 28: 'hardness5', 29: 'hardness5'}, 'train_acc': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0, 29: 1.0}, 'val_acc': {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.5, 7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5, 17: 0.5, 18: 1.0, 19: 1.0, 20: 1.0, 21: 0.5, 22: 0.5, 23: 0.5, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0, 29: 1.0}, 'standard': {0: 'hardness3', 1: 'hardness4', 2: 'hardness5', 3: 'hardness2', 4: 'hardness4', 5: 'hardness5', 6: 'hardness2', 7: 'hardness3', 8: 'hardness5', 9: 'hardness2', 10: 'hardness3', 11: 'hardness4', 12: 'hardness1', 13: 'hardness4', 14: 'hardness5', 15: 'hardness1', 16: 'hardness3', 17: 'hardness5', 18: 'hardness1', 19: 'hardness3', 20: 'hardness4', 21: 'hardness1', 22: 'hardness2', 23: 'hardness5', 24: 'hardness1', 25: 'hardness2', 26: 'hardness4', 27: 'hardness1', 28: 'hardness2', 29: 'hardness3'}, 'prediction1': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0, 10: 1.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 0.0, 28: 1.0, 29: 1.0}, 'prediction2': {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0, 10: 0.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 1.0, 28: 0.0, 29: 0.0}, 'accuracy': {0: 0.0, 1: 0.0, 2: 0.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 100.0, 12: 0.0, 13: 100.0, 14: 100.0, 15: 0.0, 16: 0.0, 17: 100.0, 18: 100.0, 19: 100.0, 20: 0.0, 21: 100.0, 22: 100.0, 23: 0.0, 24: 100.0, 25: 100.0, 26: 0.0, 27: 0.0, 28: 100.0, 29: 100.0}}



    dataframe = pd.DataFrame(dataframe)
    [confusion_matrix, true_confusion_matrix] = compute_confusion_matrix(dataframe, dataset="Own")
    accuracy = compute_accuracy(dataframe, dataset="Own")
    print("Confusion matrix: \n", confusion_matrix)
    print("True Confusion Matrix: \n", true_confusion_matrix)
    print("Average Accuracy: \n", accuracy*100)


  