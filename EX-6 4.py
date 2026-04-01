import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
print("Name:Rithanya|Roll No:24BAD132")
df=pd.read_csv(r"C:\Users\HP\Downloads\heart_stacking.csv")
df.columns=df.columns.str.strip().str.lower()
X=df[['cholesterol','maxheartrate','age']]
y=df['heartdisease']
le=LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col]=le.fit_transform(df[col])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LogisticRegression(max_iter=1000)
svm=SVC(probability=True)
dt=DecisionTreeClassifier()
lr.fit(X_train,y_train)
svm.fit(X_train,y_train)
dt.fit(X_train,y_train)
acc_lr=accuracy_score(y_test,lr.predict(X_test))
acc_svm=accuracy_score(y_test,svm.predict(X_test))
acc_dt=accuracy_score(y_test,dt.predict(X_test))
estimators=[('lr',lr),('svm',svm),('dt',dt)]
stack=StackingClassifier(estimators=estimators,final_estimator=LogisticRegression())
stack.fit(X_train,y_train)
acc_stack=accuracy_score(y_test,stack.predict(X_test))
print("LR:",acc_lr)
print("SVM:",acc_svm)
print("DT:",acc_dt)
print("Stacking:",acc_stack)
plt.figure()
plt.bar(['LR','SVM','DT','Stack'],[acc_lr,acc_svm,acc_dt,acc_stack])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()