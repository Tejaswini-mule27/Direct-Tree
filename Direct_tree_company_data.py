# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:03:06 2026

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
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

#----------------------------------
# Step 2: Load Dataset
#----------------------------------
df = pd.read_csv("C:/12-Supervise_Machine_Learning_Algo/Company_Data.csv")

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
# Step 6: Target Variable Creation (Sales Category)
#----------------------------------
bins = [0, 5, 10, 15, 20]
labels = ['Low', 'Average', 'Good', 'Better']

df['Sales_cat'] = pd.cut(df['Sales'], bins=bins, labels=labels)

print(df['Sales_cat'].value_counts())

#----------------------------------
# Step 7: Encoding Categorical Variables
#----------------------------------
le = LabelEncoder()
df['ShelveLoc'] = le.fit_transform(df['ShelveLoc'])
df['Urban'] = le.fit_transform(df['Urban'])
df['US'] = le.fit_transform(df['US'])
df['Sales_cat'] = le.fit_transform(df['Sales_cat'])

#----------------------------------
# Step 8: Outlier Detection (Boxplots)
#----------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=df.select_dtypes(include=np.number), orient='h')
plt.title("Boxplot Before Outlier Treatment")
plt.show()

#----------------------------------
# Step 9: Outlier Treatment (Winsorization)
#----------------------------------
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['CompPrice', 'Sales', 'Price']
)
df[['CompPrice','Sales','Price']] = winsor.fit_transform(df[['CompPrice','Sales','Price']])

plt.figure(figsize=(12,6))
sns.boxplot(data=df[['CompPrice','Sales','Price']], orient='h')
plt.title("Boxplot After Winsorization")
plt.show()

#----------------------------------
# Step 10: Skewness Detection
#----------------------------------
num_cols = df.select_dtypes(include=np.number).columns
skew_values = df[num_cols].apply(lambda x: skew(x))
print("\nSkewness:\n", skew_values)

#----------------------------------
# Step 11: Train-Test Split
#----------------------------------
X = df.drop(columns=['Sales','Sales_cat'])
y = df['Sales_cat']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#----------------------------------
# MODEL 1: Decision Tree (Baseline)
#----------------------------------
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

print("Decision Tree Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Decision Tree Test Accuracy :", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

#----------------------------------
# MODEL 2: Decision Tree Optimization
#----------------------------------
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [5, 10, 15]
}

grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(criterion='entropy', random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)
print("Best Parameters:", grid_dt.best_params_)

dt_opt = grid_dt.best_estimator_
y_test_pred_opt = dt_opt.predict(X_test)

print("Optimized Decision Tree Test Accuracy:", accuracy_score(y_test, y_test_pred_opt))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred_opt))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred_opt))

#----------------------------------
# MODEL 3: Random Forest
#----------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_test_pred_rf = rf.predict(X_test)

print("Random Forest Test Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred_rf))
