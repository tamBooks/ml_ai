# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Database and Cleaning
def clean_data(df, dataset_type=None):
    # Drop rows with NaN outcomes
    if dataset_type == "train":
        df = df.dropna(subset=['Genetic Disorder', 'Disorder Subclass'], how='any')

    # Feature selection and NaN handling
    rhs = df.drop(['Patient Id', 'Patient First Name', 'Family Name', "Father's name",
                   'Institute Name', 'Institute Name', 'Location of Institute', 'Status',
                   'Genetic Disorder', 'Disorder Subclass'], axis=1, errors='ignore')

    for column in rhs.select_dtypes(include='object').columns:
        rhs[column] = rhs[column].replace("-99", np.NaN)
        rhs[column] = rhs[column].fillna(rhs[column].mode().values[0])
        rhs = pd.concat([rhs, pd.get_dummies(rhs[column], drop_first=True, prefix=column)], axis=1)
        rhs = rhs.drop(columns=column)

    for column in rhs.select_dtypes(include=['float64', 'int64']):
        rhs[column] = rhs[column].replace(-99, np.NaN)
        rhs[column] = rhs[column].fillna(rhs[column].median())

    if dataset_type == "train":
        lhs = df[["Genetic Disorder", "Disorder Subclass"]]
        return lhs, rhs
    elif dataset_type == "test":
        return rhs
    else:
        raise Exception("Please define sample as 'test' or 'train'.")

# Clean database
labels, features = clean_data(pd.read_csv("../data/train.csv"), "train")
features.head(5)

# Check missing values
features.isna().sum()

# Visualization of Genetic Disorder
plt.figure(figsize=(10, 5), dpi=100)
plt.hist(labels['Genetic Disorder'])
plt.title("Distribution of Genetic Disorders")
plt.show()

# Visualization of Subclasses
plt.figure(figsize=(20, 8))
plt.hist(labels['Disorder Subclass'], color="lightblue", ec="black")
plt.title("Distribution of Disorder Subclasses")
plt.show()

# Splitting dataset
y_genetic_disorder = labels['Genetic Disorder']
y_disorder_subclass = labels['Disorder Subclass']

x_genetic_disorder_train, x_genetic_disorder_test, y_genetic_disorder_train, y_genetic_disorder_test = \
    train_test_split(features, y_genetic_disorder, test_size=0.2, random_state=42)

x_disorder_subclass_train, x_disorder_subclass_test, y_disorder_subclass_train, y_disorder_subclass_test = \
    train_test_split(features, y_disorder_subclass, test_size=0.2, random_state=42)

# Training
clf_genetic_disorder = RandomForestClassifier(max_depth=9, random_state=42)
clf_genetic_disorder.fit(x_genetic_disorder_train, y_genetic_disorder_train)

clf_disorder_subclass = RandomForestClassifier(max_depth=9, random_state=42)
clf_disorder_subclass.fit(x_disorder_subclass_train, y_disorder_subclass_train)

# Model evaluation
print(f"Genetic Disorder Accuracy on Train Set: {clf_genetic_disorder.score(x_genetic_disorder_train, y_genetic_disorder_train)}")
print(f"Disorder Subclass Accuracy on Train Set: {clf_disorder_subclass.score(x_disorder_subclass_train, y_disorder_subclass_train)}")

print(f"Genetic Disorder Accuracy on Test Set: {clf_genetic_disorder.score(x_genetic_disorder_test, y_genetic_disorder_test)}")
print(f"Disorder Subclass Accuracy on Test Set: {clf_disorder_subclass.score(x_disorder_subclass_test, y_disorder_subclass_test)}")

# Additional evaluation metrics
y_genetic_disorder_pred = clf_genetic_disorder.predict(x_genetic_disorder_test)
y_disorder_subclass_pred = clf_disorder_subclass.predict(x_disorder_subclass_test)

print("Genetic Disorder Classification Report:")
print(classification_report(y_genetic_disorder_test, y_genetic_disorder_pred))

print("Disorder Subclass Classification Report:")
print(classification_report(y_disorder_subclass_test, y_disorder_subclass_pred, zero_division=1)))
