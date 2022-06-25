import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

#reading data and labels file
data=pd.read_csv("data/data.csv")
labels=pd.read_csv("data/labels.csv")

#Dropping the column that is also present in the labels file
feature_space = data.drop('Sample', axis = 1)
#Representing every disease type as a class
feature_class = labels['disease_type']

#Splitting data to test-train data sets %75 for train %25 for test
training_set, test_set, class_set, test_class_set = train_test_split(feature_space, feature_class, test_size = 0.25, random_state = 42)

class_set = class_set.values.ravel()
test_class_set = test_class_set.values.ravel()

#Normaliazing the data
scaler = StandardScaler()
scaler.fit(training_set)

training_set=scaler.transform(test_set)



