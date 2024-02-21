# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Database and Cleaning
# Define cleaning function to apply to both testing and training data
def clean_data(df, set = None):
       
       # Drop row if outcome is NaN, can't train on these
       if set == "train": df = df.dropna(subset=['Genetic Disorder', 'Disorder Subclass'], how='any')

       rhs = df.drop(['Patient Id', 'Patient First Name', 'Family Name', "Father's name",
              'Institute Name', 'Institute Name', 'Location of Institute', 'Status',
              'Genetic Disorder', 'Disorder Subclass'
              ], axis = 1, errors = 'ignore')
       
       # Fill NaN with mode for categorical columns, median for numeric columns
       for i in rhs.select_dtypes(include='object').columns:

              # Unfortunately, the data we chose has a test set deliberately corrupted with -99 values...
              rhs[i] = rhs[i].replace("-99", np.NaN)
              rhs[i] = rhs[i].fillna(rhs[i].mode().values[0])

              # Create dummies for categorical columns (one hot encoding)
              rhs = pd.concat([rhs, pd.get_dummies(rhs[i], drop_first=True, prefix=i)], axis = 1)
              rhs = rhs.drop(columns=i)

       for i in rhs.select_dtypes(include=['float64', 'int64']):

              # Unfortunately, the data we chose has a test set deliberately corrupted with -99 values...
              rhs[i] = rhs[i].replace(-99, np.NaN)
              rhs[i] = rhs[i].fillna(rhs[i].median())

       # Output is train/test dependant
       if set == "train":

              lhs = df[["Genetic Disorder", "Disorder Subclass"]]
       
              return lhs, rhs
       
       if set == "test": return rhs

       else: raise Exception("Please define sample as 'test' or 'train'.") 

# Clean database
labels, x = clean_data(pd.read_csv("../data/train.csv"), "train")
x.head(5)

# Check to make sure we've recoded all NAs
x.isna().sum()

# Visualization of Genetic Disorder
plt.figure(figsize=(10,5),dpi = 100)
plt.hist(labels['Genetic Disorder'])

# Visualization of Subclasses
plt.figure(figsize=(20,8))
plt.hist(labels['Disorder Subclass'], color = "lightblue", ec="black")

# Spliting of dataset
y1 = labels['Genetic Disorder']
y2 = labels['Disorder Subclass']

x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, test_size=0.2)

# Training 
clf1 = RandomForestClassifier(max_depth=9, random_state=0)
clf1.fit(x1_train, y1_train)

clf2 = RandomForestClassifier(max_depth=9, random_state=0)
clf2.fit(x2_train, y2_train)

clf1.score(x1_train, y1_train)
clf2.score(x2_train, y2_train)

# Testing 
clf1.score(x1_test, y1_test)
clf2.score(x2_test, y2_test)


























