# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# Load dataset
data = pd.read_csv(r"C:\Users\HP\Downloads\diabetes_bagging.csv")
# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# -------------------------------
# 1. Decision Tree Model
# -------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
# -------------------------------
# 2. Bagging Classifier
# -------------------------------
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bag.fit(X_train, y_train)
y_pred_bag = bag.predict(X_test)
bag_acc = accuracy_score(y_test, y_pred_bag)

# -------------------------------
# Accuracy Comparison
# -------------------------------
print("RITHANYA.G 24BAD132")
print("Decision Tree Accuracy:", dt_acc)
print("Bagging Accuracy:", bag_acc)
# Bar Graph
models = ["Decision Tree", "Bagging"]
accuracies = [dt_acc, bag_acc]
plt.figure()
plt.bar(models, accuracies)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.show()
# -------------------------------
# Confusion Matrix (Bagging)
# -------------------------------
cm = confusion_matrix(y_test, y_pred_bag)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Bagging")
plt.show()