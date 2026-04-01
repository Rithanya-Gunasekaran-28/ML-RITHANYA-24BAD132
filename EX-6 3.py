import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
print("Name:Rithanya|Roll No:24BAD132")
df=pd.read_csv(r"C:\Users\HP\Downloads\income_random_forest.csv")
df.columns=df.columns.str.strip().str.lower().str.replace(' ','-').str.replace('_','-')
print("Columns:",df.columns)
# adjust column names based on your dataset
# MODIFY these names if needed after seeing output above
cols=[]
if 'age' in df.columns: cols.append('age')
if 'education' in df.columns: cols.append('education')
elif 'education-num' in df.columns: cols.append('education-num')
if 'occupation' in df.columns: cols.append('occupation')
elif 'workclass' in df.columns: cols.append('workclass')
if 'hours-per-week' in df.columns: cols.append('hours-per-week')
if 'income' in df.columns: target='income'
elif 'salary' in df.columns: target='salary'
elif 'income-level' in df.columns: target='income-level'
cols.append(target)
df=df[cols].dropna()
le=LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col]=le.fit_transform(df[col])
X=df.drop(target,axis=1)
y=df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
n_trees=[10,50,100,150,200]
acc_list=[]
for n in n_trees:
    rf=RandomForestClassifier(n_estimators=n,random_state=42)
    rf.fit(X_train,y_train)
    acc_list.append(accuracy_score(y_test,rf.predict(X_test)))
best_n=n_trees[acc_list.index(max(acc_list))]
model=RandomForestClassifier(n_estimators=best_n,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("BestTrees:",best_n)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
plt.figure()
plt.plot(n_trees,acc_list,marker='o')
plt.xlabel("Trees")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Trees")
plt.show()
plt.figure()
plt.bar(X.columns,model.feature_importances_)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()