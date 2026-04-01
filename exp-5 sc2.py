print("RITHANYA.G 24BAD132")

# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===== LOAD DATASET AUTOMATICALLY FROM FOLDER =====

folder_path = r"C:\Users\HP\Downloads\archive (18)"

# Find CSV file inside folder
files = os.listdir(folder_path)
csv_file = [file for file in files if file.endswith(".csv")][0]

df = pd.read_csv(os.path.join(folder_path, csv_file))

print("Loaded File:", csv_file)
print("\nFirst 5 rows:")
print(df.head())

# ===== HANDLE MISSING VALUES =====

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ===== SELECT FEATURES =====

features = ['ApplicantIncome', 'LoanAmount',
            'Credit_History', 'Education', 'Property_Area']

X = df[features].copy()
y = df['Loan_Status']

# ===== ENCODING =====

le = LabelEncoder()

for col in X.select_dtypes(include='object').columns:
    X[col] = le.fit_transform(X[col])

y = le.fit_transform(y)

# ===== TRAIN TEST SPLIT =====

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ===== TRAIN MODEL =====

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# ===== PREDICT =====

y_pred = dt.predict(X_test)

# ===== EVALUATION =====

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ===== CONFUSION MATRIX PLOT (No Seaborn) =====

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.colorbar()
plt.show()

# ===== TREE VISUALIZATION =====

plt.figure(figsize=(15,8))
plot_tree(dt,
          feature_names=features,
          class_names=["Rejected", "Approved"],
          filled=True)
plt.title("Decision Tree Structure")
plt.show()

# ===== FEATURE IMPORTANCE =====

importance = pd.DataFrame({
    "Feature": features,
    "Importance": dt.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance)

plt.figure()
plt.bar(importance["Feature"], importance["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()