
"""
Key Changes Made:

 1. Colormap Implementation: Added plt.cm.plasma(np.linspace(0.2, 0.8, len(df))). 
    This automatically generates a sequence of colors. 0.2 to 0.8 ensures 
    we stay within the vibrant range of the spectrum.
 2. Bordering: Added edgecolor='black' to the bars to make the colors pop against the white background.
 3. Dynamic Labeling: Added code to print the exact importance value (e.g., 0.270) next to each bar, 
    which is a standard practice in medical reporting for precision.
 4. Sorting: The bars are sorted in ascending order so the "Health Drivers" (most important) are always 
    at the top with the brightest color.

In a healthcare setting, this gradient helps clinicians immediately identify Glucose 
    and BMI as the primary indicators they should focus on during patient consultation.
    
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD DATASET
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'Pedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# 2. DATA CLEANING
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].median())

# 3. TRAIN MODEL
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. PREPARE COLOR VARIATION DATA
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# 5. PLOTTING WITH GRADIENT COLORS
# Using the 'plasma' colormap to create a range from purple to yellow
colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(feature_importance_df)))

plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance_df['Feature'], 
                feature_importance_df['Importance'], 
                color=colors, 
                edgecolor='black', 
                alpha=0.8)

# Adding visual polish
plt.xlabel('Influence on Prediction (Feature Importance Score)')
plt.ylabel('Clinical Biomarkers')
plt.title('Healthcare Risk Factors: AI Feature Importance Gradient')
plt.grid(axis='x', linestyle='--', alpha=0.3)

# Adding text labels on the bars
for bar in bars:
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()