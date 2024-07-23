import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from lib_pre_data import *
from sklearn.metrics import classification_report, confusion_matrix

data_train = pd.read_csv('Data/train.csv')
data_test = pd.read_csv('Data/test.csv')
data_results = pd.read_csv('Data/gender_submission.csv')

data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch'] + 1
data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch'] + 1

data_train['Title'] = data_train['Name'].apply(search_title_name)
data_test['Title'] = data_test['Name'].apply(search_title_name)
title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master", "Dr": "Rare", "Rev": "Rare",
    "Col": "Rare", "Major": "Rare", "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss",
    "Lady": "Rare", "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Mme": "Mrs",
    "Capt": "Rare", "Sir": "Rare"
}

data_train['Title'] = data_train['Title'].map(title_mapping)
data_test['Title'] = data_test['Title'].map(title_mapping)

imputer = SimpleImputer(strategy = 'median')
data_train['Age'] = imputer.fit_transform(data_train[['Age']])
data_test['Age'] = imputer.transform(data_test[['Age']])

data_train, data_test = encoder_colum(data_train, data_test, 'Sex')
data_train, data_test = encoder_colum(data_train, data_test, 'Title')

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Age', 'FamilySize', 'Title']
X_train = data_train[features]
X_test = data_test[features]
y_train = data_train['Survived']
test_value = data_results.iloc[:,-1].values
choose_models(X_train,y_train,X_test,test_value)

classifier = RandomForestClassifier(n_estimators = 200,max_features = 'sqrt',min_samples_leaf = 3,min_samples_split = 3,max_depth = 3, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)
predictions_value = classifier.predict(X_test)

compare = pd.DataFrame({'Predictions': predictions_value, 'Actual': test_value})
correct_predictions = (compare['Predictions'] == compare['Actual']).sum()
incorrect_predictions = (compare['Predictions'] != compare['Actual']).sum()
labels = ['Correct Predictions', 'Incorrect Predictions']
values = [correct_predictions, incorrect_predictions]
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['pink', 'blue'])
plt.title('Comparison of Predictions')
plt.ylabel('Number of Predictions')
plt.show()