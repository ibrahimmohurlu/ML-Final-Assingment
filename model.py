import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

#reading data and labels file
data = pd.read_csv("data/data.csv")
labels = pd.read_csv("data/labels.csv")

#Dropping the column that is also present in the labels file
feature_space = data.drop('Sample', axis = 1)
#Representing every disease type as a class
feature_class = labels['disease_type']

#Splitting data to test-train data sets %75 for train %25 for test
training_set, test_set, class_set, test_class_set = train_test_split(feature_space, feature_class, test_size = 0.25, random_state = 42)

#Tuple that will be passed as a parameter while creating MLPClassifier
hidden_layers=(100,80,40)

# Creating 4 different neural network to see the how different activation functions change the performance
# MLPClassifier can take 4 different parameter as activation function these are {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
mlp_1 = MLPClassifier(activation = 'identity', hidden_layer_sizes = hidden_layers, max_iter = 100, random_state = None, solver = 'lbfgs')
mlp_2 = MLPClassifier(activation = 'logistic', hidden_layer_sizes = hidden_layers, max_iter = 100, random_state = None, solver = 'lbfgs')
mlp_3 = MLPClassifier(activation = 'tanh', hidden_layer_sizes = hidden_layers, max_iter = 100, random_state = None, solver = 'lbfgs')
mlp_4 = MLPClassifier(activation = 'relu', hidden_layer_sizes = hidden_layers, max_iter = 100, random_state = None, solver = 'lbfgs')

# Fitting training sets to classifiers
mlp_1.fit(training_set, class_set)
mlp_2.fit(training_set, class_set)
mlp_3.fit(training_set, class_set)
mlp_4.fit(training_set, class_set)

# Function to calculate performance measures it will be used later
def calculate_results(classifier):
    #Getting predictions of our model
    predictions = classifier.predict(test_set)
    #Calculating Precision, Recall and F2 measure formulas
    #Precision = TP / (TP + FP)
    precision = precision_score(test_class_set, predictions, average = None)

    #Recall = TP / (TP + FN)
    recall = recall_score(test_class_set, predictions, average = None)

    #F2 measure = 5 * precision * recall / (4 * precision + recall)
    F2_measure = 5 * precision * recall / (4 * precision + recall)
    #Collecting all the measures inside of a dictionary and returning it
    results = {
        'activation_function':classifier.activation,
        'precision':precision,
        'recall':recall,
        'F2':F2_measure
    }
    return results

# Using dictionary to store precision, recall and F2 measures
results={
    'mlp_1':calculate_results(mlp_1),
    'mlp_2':calculate_results(mlp_2),
    'mlp_3':calculate_results(mlp_3),
    'mlp_4':calculate_results(mlp_4)
}

#printing performance measures
for key, value in results.items():
    print(f"{key}({value['activation_function']}):")
    print(f"Precision = {value['precision']}")
    print(f"Recall = {value['recall']}")
    print(f"F2 = {value['F2']}\n")




