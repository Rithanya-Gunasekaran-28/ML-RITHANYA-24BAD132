import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
print("Name: Rithanya Roll No: 24BAD132")
data=pd.read_csv(r"C:\Users\HP\Downloads\churn_boosting.csv")
le=LabelEncoder()
for col in data.columns:
    if data[col].dtype=='object':
        data[col]=le.fit_transform(data[col])
X=data.drop("Churn",axis=1)
y=data["Churn"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
ada=AdaBoostClassifier(n_estimators=50,random_state=42)
ada.fit(X_train,y_train)
y_pred_ada=ada.predict(X_test)
y_prob_ada=ada.predict_proba(X_test)[:,1]
ada_acc=accuracy_score(y_test,y_pred_ada)
gb=GradientBoostingClassifier(n_estimators=100,random_state=42)
gb.fit(X_train,y_train)
y_pred_gb=gb.predict(X_test)
y_prob_gb=gb.predict_proba(X_test)[:,1]
gb_acc=accuracy_score(y_test,y_pred_gb)
print("AdaBoost Accuracy:",ada_acc)
print("Gradient Boosting Accuracy:",gb_acc)
models=["AdaBoost","Gradient Boosting"]
accuracies=[ada_acc,gb_acc]
plt.figure()
plt.bar(models,accuracies)
plt.title("Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()
fpr_ada,tpr_ada,_=roc_curve(y_test,y_prob_ada)
fpr_gb,tpr_gb,_=roc_curve(y_test,y_prob_gb)
roc_auc_ada=auc(fpr_ada,tpr_ada)
roc_auc_gb=auc(fpr_gb,tpr_gb)
plt.figure()
plt.plot(fpr_ada,tpr_ada,label="AdaBoost (AUC=%.2f)"%roc_auc_ada)
plt.plot(fpr_gb,tpr_gb,label="Gradient Boosting (AUC=%.2f)"%roc_auc_gb)
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
importances=gb.feature_importances_
features=X.columns
plt.figure()
plt.barh(features,importances)
plt.title("Feature Importance (Gradient Boosting)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()