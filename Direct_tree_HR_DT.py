# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:09:42 2026

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")

#----------------------------------
# Step 2: Load Dataset
#----------------------------------
df = pd.read_csv("C:/Assignment/Direct_tree/HR_DT.csv")

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
# Step 6: Encoding Categorical Variables
#----------------------------------
le = LabelEncoder()
df['Position'] = le.fit_transform(df['Position'])

#----------------------------------
# Step 7: Exploratory Data Analysis
#----------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x='no of Years of Experience of employee',
                y='monthly income of employee',
                hue='Position',
                data=df,
                palette='tab10')
plt.title("Experience vs Monthly Income by Position")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Position', y='monthly income of employee', data=df)
plt.title("Income Distribution by Position")
plt.show()

#----------------------------------
# Step 8: Feature & Target Split
#----------------------------------
X = df[['Position','no of Years of Experience of employee']]
y = df['monthly income of employee']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#----------------------------------
# MODEL 1: Decision Tree Regressor
#----------------------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print("Decision Tree R2 Score:", r2_score(y_test, y_pred_dt))

#----------------------------------
# MODEL 2: Random Forest Regressor
#----------------------------------
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))

#----------------------------------
# FINAL BUSINESS INTERPRETATION
#----------------------------------
"""
Decision Tree:
- Captures non-linear relationships between experience and income.
- May overfit small datasets.

Random Forest:
- More stable and generalizable.
- Provides better accuracy and lower error.

Business Value:
- Helps HR predict employee salary based on role and experience.
- Supports workforce planning and compensation benchmarking.
"""
