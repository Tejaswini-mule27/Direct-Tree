# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:21:42 2026

@author: YADNYESH
"""



#----------------------------------
# Step 1: Import Required Libraries
#----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

#----------------------------------
# Step 2: Load Dataset
#----------------------------------
df = pd.read_csv("C:/Assignments/Direct_tree/Diabetes.csv")

print("Initial Shape:", df.shape)
print(df.head())

#----------------------------------
# Step 3: Data Type Check
#----------------------------------
print(df.dtypes)

#----------------------------------
# Step 4: Missing Value Analysis
#----------------------------------
print("Missing values:\n", df.isnull().sum())

#----------------------------------
# Step 5: Duplicate Removal
#----------------------------------
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

#----------------------------------
# Step 6: Target Variable Encoding
#----------------------------------
# Encode Class variable (YES/NO → 1/0)
le = LabelEncoder()
df['Class variable'] = le.fit_transform(df['Class variable'])

print(df['Class variable'].value_counts())

#----------------------------------
# Step 7: Exploratory Data Analysis
#----------------------------------
plt.figure(figsize=(8,6))
sns.countplot(x='Class variable', data=df, palette='Set2')
plt.title("Diabetes Distribution (YES vs NO)")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Class variable', y='Plasma glucose concentration', data=df)
plt.title("Plasma Glucose vs Diabetes")
plt.show()

#----------------------------------
# Step 8: Feature & Target Split
#----------------------------------
X = df.drop(columns=['Class variable'])
y = df['Class variable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#----------------------------------
# MODEL 1: Decision Tree Classifier
#----------------------------------
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

#----------------------------------
# MODEL 2: Random Forest Classifier
#----------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

#----------------------------------
# FINAL BUSINESS INTERPRETATION
#----------------------------------
"""
Decision Tree:
- Provides simple rules for predicting diabetes.
- May overfit small datasets.

Random Forest:
- More robust and generalizable.
- Better accuracy and balanced classification.

Business Value:
- Helps healthcare professionals identify patients at risk of diabetes.
- Supports preventive care and early intervention.
- Reduces long-term healthcare costs by flagging high-risk individuals.
"""
