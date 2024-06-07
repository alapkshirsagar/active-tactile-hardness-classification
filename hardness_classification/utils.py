import pandas as pd
import numpy as np
from itertools import combinations, permutations

# True hardness values: 
#Deform level 1 (least soft): 0.44
#Deform level 2: 0.68
#Deform level 3: 0.79
#Deform level 4: 1.02
#Deform level 5 (most soft): 1.13
# Sample hardness values for 5 objects
hardness_values = {"hardness1":0.44, 
                   "hardness2":0.68, 
                   "hardness3":0.79, 
                   "hardness4":1.02, 
                   "hardness5":1.13}  # Replace with actual hardness values


# Generate all combinations of the 5 objects taken 2 at a time for comparison
comparison_pairs = list(combinations(hardness_values, 2))

# List to hold the triplets
comparison_triplets = []

# Generate triplets
for comparison_pair in comparison_pairs:
    for standard in hardness_values:
        if standard not in comparison_pair:
            comparison_triplets.append((*comparison_pair, standard))

print("Shape of comparison_triplets: ", len(comparison_triplets))
print("Comparison triplets:", comparison_triplets)

# Function to return similarity results in train_2_classify_three scenario
def compare_hardness(comparison1, comparison2, standard):
    closer_to_comparison1 = abs(hardness_values[comparison1] - hardness_values[standard]) < abs(hardness_values[comparison2] - hardness_values[standard])
    return (1 if closer_to_comparison1 else 0, 0 if closer_to_comparison1 else 1)


def compute_confusion_matrix(dataframe):
    confusion_matrix = pd.DataFrame(0.0, 
                                    index=['hardness1', 'hardness2', 'hardness3', 'hardness4', 'hardness5'], 
                                    columns=['hardness1', 'hardness2', 'hardness3', 'hardness4', 'hardness5'])
    true_confusion_matrix = pd.DataFrame(0.0, 
                                    index=['hardness1', 'hardness2', 'hardness3', 'hardness4', 'hardness5'], 
                                    columns=['hardness1', 'hardness2', 'hardness3', 'hardness4', 'hardness5'])
    for triplet in comparison_triplets:
        comparison1_object = triplet[0]
        comparison2_object = triplet[1]
        standard_object = triplet[2]
        [true_prediction1, true_prediction2] = compare_hardness(comparison1_object, comparison2_object, standard_object)
        trial_row = dataframe.loc[
            (dataframe['standard'] == standard_object) &
            (dataframe['comparison1'] == comparison1_object) &
            (dataframe['comparison2'] == comparison2_object)
        ]
        confusion_matrix.at[standard_object, comparison1_object] += trial_row['prediction1'].iat[0]
        confusion_matrix.at[standard_object, comparison2_object]+= trial_row['prediction2'].iat[0]

        true_confusion_matrix.at[standard_object, comparison1_object] += true_prediction1*(trial_row['prediction1'].iat[0] + trial_row['prediction2'].iat[0])
        true_confusion_matrix.at[standard_object, comparison2_object] += true_prediction2*(trial_row['prediction1'].iat[0] + trial_row['prediction2'].iat[0])
    # Normalize the rows of the confusion matrix
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)
    true_confusion_matrix = true_confusion_matrix.div(true_confusion_matrix.sum(axis=1), axis=0)
    return confusion_matrix, true_confusion_matrix


def compute_acc(dataframe):
    accuracy = []
    for triplet in comparison_triplets:
        comparison1_object = triplet[0]
        comparison2_object = triplet[1]
        standard_object = triplet[2]
        [true_prediction1, true_prediction2] = compare_hardness(comparison1_object, comparison2_object, standard_object)
        trial_row = dataframe.loc[
            (dataframe['standard'] == standard_object) &
            (dataframe['comparison1'] == comparison1_object) &
            (dataframe['comparison2'] == comparison2_object)
        ]
        accuracy.append(1-abs(true_prediction1-trial_row['prediction1']/(trial_row['prediction1']+trial_row['prediction2'])))
        print(accuracy[-1])
    
    
    return np.mean(np.array(accuracy))



if __name__ == "__main__":
    # Your DataFrame
    dataframe = {'comparison1': {0: 'hardness1', 1: 'hardness1', 2: 'hardness1', 3: 'hardness1', 4: 'hardness1', 5: 'hardness1', 6: 'hardness1', 7: 'hardness1', 8: 'hardness1', 9: 'hardness1', 10: 'hardness1', 11: 'hardness1', 12: 'hardness2', 13: 'hardness2', 14: 'hardness2', 15: 'hardness2', 16: 'hardness2', 17: 'hardness2', 18: 'hardness2', 19: 'hardness2', 20: 'hardness2', 21: 'hardness3', 22: 'hardness3', 23: 'hardness3', 24: 'hardness3', 25: 'hardness3', 26: 'hardness3', 27: 'hardness4', 28: 'hardness4', 29: 'hardness4'}, 'comparison2': {0: 'hardness2', 1: 'hardness2', 2: 'hardness2', 3: 'hardness3', 4: 'hardness3', 5: 'hardness3', 6: 'hardness4', 7: 'hardness4', 8: 'hardness4', 9: 'hardness5', 10: 'hardness5', 11: 'hardness5', 12: 'hardness3', 13: 'hardness3', 14: 'hardness3', 15: 'hardness4', 16: 'hardness4', 17: 'hardness4', 18: 'hardness5', 19: 'hardness5', 20: 'hardness5', 21: 'hardness4', 22: 'hardness4', 23: 'hardness4', 24: 'hardness5', 25: 'hardness5', 26: 'hardness5', 27: 'hardness5', 28: 'hardness5', 29: 'hardness5'}, 'val_loss': {0: 0.6958777904510498, 1: 0.6958777904510498, 2: 0.6958777904510498, 3: 0.6966618299484253, 4: 0.6966618299484253, 5: 0.6966618299484253, 6: 0.6920482516288757, 7: 0.6920482516288757, 8: 0.6920482516288757, 9: 0.6983448266983032, 10: 0.6983448266983032, 11: 0.6983448266983032, 12: 0.6955512762069702, 13: 0.6955512762069702, 14: 0.6955512762069702, 15: 0.6961017847061157, 16: 0.6961017847061157, 17: 0.6961017847061157, 18: 0.7262738943099976, 19: 0.7262738943099976, 20: 0.7262738943099976, 21: 0.6995419263839722, 22: 0.6995419263839722, 23: 0.6995419263839722, 24: 1.1004825830459595, 25: 1.1004825830459595, 26: 1.1004825830459595, 27: 0.6968526244163513, 28: 0.6968526244163513, 29: 0.6968526244163513}, 'val_acc': {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.550000011920929, 7: 0.550000011920929, 8: 0.550000011920929, 9: 0.4000000059604645, 10: 0.4000000059604645, 11: 0.4000000059604645, 12: 0.44999998807907104, 13: 0.44999998807907104, 14: 0.44999998807907104, 15: 0.5, 16: 0.5, 17: 0.5, 18: 0.3499999940395355, 19: 0.3499999940395355, 20: 0.3499999940395355, 21: 0.4000000059604645, 22: 0.4000000059604645, 23: 0.4000000059604645, 24: 0.30000001192092896, 25: 0.30000001192092896, 26: 0.30000001192092896, 27: 0.4000000059604645, 28: 0.4000000059604645, 29: 0.4000000059604645}, 'standard': {0: 'hardness3', 1: 'hardness4', 2: 'hardness5', 3: 'hardness2', 4: 'hardness4', 5: 'hardness5', 6: 'hardness2', 7: 'hardness3', 8: 'hardness5', 9: 'hardness2', 10: 'hardness3', 11: 'hardness4', 12: 'hardness1', 13: 'hardness4', 14: 'hardness5', 15: 'hardness1', 16: 'hardness3', 17: 'hardness5', 18: 'hardness1', 19: 'hardness3', 20: 'hardness4', 21: 'hardness1', 22: 'hardness2', 23: 'hardness5', 24: 'hardness1', 25: 'hardness2', 26: 'hardness4', 27: 'hardness1', 28: 'hardness2', 29: 'hardness3'}, 'prediction1': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0, 29: 1.0}, 'prediction2': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 0.0, 28: 0.0, 29: 0.0}, 'accuracy': {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 0.0, 7: 100.0, 8: 100.0, 9: 100.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 100.0, 14: 100.0, 15: 100.0, 16: 100.0, 17: 0.0, 18: 100.0, 19: 100.0, 20: 0.0, 21: 100.0, 22: 100.0, 23: 0.0, 24: 100.0, 25: 100.0, 26: 0.0, 27: 100.0, 28: 100.0, 29: 100.0}}

    dataframe = pd.DataFrame(dataframe)
    [confusion_matrix, true_confusion_matrix] = compute_confusion_matrix(dataframe)
    accuracy = compute_acc(dataframe)
    print("Confusion matrix: \n", confusion_matrix)
    print("True Confusion Matrix: \n", true_confusion_matrix)
    print("Average Accuracy: \n", accuracy*100)


  