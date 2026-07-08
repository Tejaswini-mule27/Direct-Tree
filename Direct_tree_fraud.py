# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:14:17 2026

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
df = pd.read_csv("C:/12-Supervise_Machine_Learning_Algo/Fraud_check.csv")

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
# Step 6: Target Variable Creation
#----------------------------------
# Business Rule: Taxable Income <= 30000 → Fraud Risk = 1, else 0
df['Fraud'] = np.where(df['Taxable.Income'] <= 30000, 1, 0)

print(df['Fraud'].value_counts())

#----------------------------------
# Step 7: Encoding Categorical Variables
#----------------------------------
le = LabelEncoder()
df['Undergrad'] = le.fit_transform(df['Undergrad'])
df['Marital.Status'] = le.fit_transform(df['Marital.Status'])
df['Urban'] = le.fit_transform(df['Urban'])

#----------------------------------
# Step 8: Exploratory Data Analysis
#----------------------------------
plt.figure(figsize=(8,6))
sns.countplot(x='Fraud', data=df, palette='Set2')
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Fraud', y='Taxable.Income', data=df)
plt.title("Taxable Income vs Fraud")
plt.show()

#----------------------------------
# Step 9: Feature & Target Split
#----------------------------------
X = df.drop(columns=['Fraud'])
y = df['Fraud']

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
- Provides simple rules for fraud detection.
- May overfit small datasets.

Random Forest:
- More robust and generalizable.
- Better accuracy and balanced classification.

Business Value:
- Helps identify potential fraud cases based on taxable income and demographics.
- Supports compliance and risk management.
- Reduces financial losses by flagging suspicious records.
"""
