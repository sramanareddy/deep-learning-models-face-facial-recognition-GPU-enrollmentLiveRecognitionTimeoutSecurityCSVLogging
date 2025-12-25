import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. LOAD DATASET (Pima Indians Diabetes)
# Note: You can download this CSV from Kaggle or UC Irvine Repository
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
url = 'data/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'Pedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# 2. DATA CLEANING (Healthcare Specific)
# In this dataset, 0 is used to represent missing data in some columns. 
# We replace 0s with the median value of that column.
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].median())

# 3. TRAIN/TEST SPLIT
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE MODEL (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. PREDICTIVE ANALYTICS & INTERPRETABILITY
print("📋 Clinical Prediction Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. HOW THE MODEL DECIDES (Feature Importance)
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n🔍 How the AI Ranks Risk Factors:")
print(feature_importance_df)

# Plotting the "Decision Drivers"
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Influence on Prediction')
plt.title('Which Biomarkers Drive Diabetes Risk?')
plt.gca().invert_yaxis()
plt.show()