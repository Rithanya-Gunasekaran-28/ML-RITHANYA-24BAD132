

print("RITHANYA.G 24BAD132")

# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset (Change path if needed)
df = pd.read_csv(r"C:\Users\HP\Downloads\archive (17)\breast-cancer.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Select Required Features
features = ['radius_mean', 'texture_mean',
            'perimeter_mean', 'area_mean',
            'smoothness_mean']

X = df[features]
y = df['diagnosis']

# Encode Target Labels
le = LabelEncoder()
y = le.fit_transform(y)   # Benign=0, Malignant=1

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train KNN Classifier (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict Diagnosis Labels
y_pred = knn.predict(X_test)

# Evaluate Performance
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Confusion Matrix Visualization (Matplotlib Version)
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

# Identify Misclassified Cases
misclassified = np.where(y_test != y_pred)
print("\nMisclassified Indices:", misclassified)
print("Number of Misclassified Cases:", len(misclassified[0]))

# Experiment with Different K Values
accuracy_list = []

for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_k = model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred_k))

# Plot Accuracy vs K
plt.figure()
plt.plot(range(1, 21), accuracy_list)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()

# Decision Boundary (Using 2 Features)
X2 = df[['radius_mean', 'texture_mean']]
X2_scaled = scaler.fit_transform(X2)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2_scaled, y, test_size=0.2, random_state=42)

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train2, y_train2)

# Create mesh grid
h = 0.02
x_min, x_max = X_train2[:, 0].min() - 1, X_train2[:, 0].max() + 1
y_min, y_max = X_train2[:, 1].min() - 1, X_train2[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train2)
plt.xlabel("Radius")
plt.ylabel("Texture")
plt.title("Decision Boundary (K=5)")
plt.show()