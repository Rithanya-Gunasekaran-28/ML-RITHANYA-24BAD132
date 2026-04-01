#Name:Rithanya|Roll No:24BAD132

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,precision_recall_curve,auc

#Load dataset
df=pd.read_csv(r"C:\Users\HP\Downloads\archive (23)\Oral cancer Dataset 2.0")

print("Columns:",df.columns)

#Target column
if 'Fraud' in df.columns:target='Fraud'
elif 'Class' in df.columns:target='Class'
else:target=df.columns[-1]

X=df.drop(target,axis=1).values
y=df[target].values

#Check imbalance
print("Before SMOTE:\n",pd.Series(y).value_counts())

plt.figure()
pd.Series(y).value_counts().plot(kind='bar')
plt.title("Before SMOTE")
plt.show()

#Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#Manual SMOTE function
def manual_smote(X,y,k=5):
    X_min=X[y==1]
    X_maj=X[y==0]

    n_samples=len(X_maj)-len(X_min)

    nbrs=NearestNeighbors(n_neighbors=k).fit(X_min)
    synthetic=[]

    for i in range(n_samples):
        idx=np.random.randint(0,len(X_min))
        x=X_min[idx]
        _,indices=nbrs.kneighbors([x])
        nn=X_min[np.random.choice(indices[0][1:])]
        gap=np.random.rand()
        new_sample=x+gap*(nn-x)
        synthetic.append(new_sample)

    X_syn=np.array(synthetic)
    y_syn=np.ones(len(X_syn))

    X_new=np.vstack((X,X_syn))
    y_new=np.hstack((y,y_syn))

    return X_new,y_new

#Apply manual SMOTE
X_train_sm,y_train_sm=manual_smote(X_train,y_train)

print("After SMOTE:\n",pd.Series(y_train_sm).value_counts())

plt.figure()
pd.Series(y_train_sm).value_counts().plot(kind='bar')
plt.title("After SMOTE")
plt.show()

#Model BEFORE SMOTE
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("\nBefore SMOTE:\n",classification_report(y_test,y_pred))

#PR curve before
y_prob=model.predict_proba(X_test)[:,1]
p,r,_=precision_recall_curve(y_test,y_prob)
auc1=auc(r,p)

plt.figure()
plt.plot(r,p)
plt.title("PR Curve Before SMOTE")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#Model AFTER SMOTE
model_sm=LogisticRegression(max_iter=1000)
model_sm.fit(X_train_sm,y_train_sm)
y_pred_sm=model_sm.predict(X_test)

print("\nAfter SMOTE:\n",classification_report(y_test,y_pred_sm))

#PR curve after
y_prob_sm=model_sm.predict_proba(X_test)[:,1]
p2,r2,_=precision_recall_curve(y_test,y_prob_sm)
auc2=auc(r2,p2)

plt.figure()
plt.plot(r2,p2)
plt.title("PR Curve After SMOTE")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#Comparison
print("\nPR-AUC Before:",auc1)
print("PR-AUC After:",auc2)