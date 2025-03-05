# Import the Library
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Import the Dataset
Dataset = pd.read_csv('creditcard.csv')
pd.options.display.max_columns = None
print("----------------------------Top 5 Feature--------------------------\n",Dataset.head())
print("----------------------------Last 5 Feature--------------------------\n",Dataset.tail())
# Understand the Dataset and Handle the data set

print(Dataset.shape) #(284807, 31)
print("Number of Rows", Dataset.shape[0]) #284807
print("number of columns", Dataset.shape[1]) #31
print(Dataset.info())
print("Missing Val", Dataset.isnull().sum())
from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
Dataset['Amount'] = sc.fit_transform(pd.DataFrame(Dataset['Amount']))
print("----------------------------After Standard Scaler--------------------------\n",Dataset.head())
Dataset = Dataset.drop('Time', axis=1)
print(Dataset.shape) #(284807, 30)
print(Dataset.duplicated().any()) #True
Dataset = Dataset.drop_duplicates()
print("After deleting Duplicate values", Dataset.shape) #(275663, 30)
# EDA on imTarget Variable
print(Dataset['Class'].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x ='Class', data= Dataset)
plt.show()
print("--------------------------------------Handle imbalance dataset----------------------------------------------")
# balance the imbalnance dataset with under and over sampling
# 1- UnderSampling the Dataset
normal = Dataset[Dataset['Class'] == 0]
print(normal.shape)
fraud = Dataset[Dataset['Class'] == 1]
print(fraud.shape)
normal_sample = normal.sample(n=473)
print(normal_sample.shape)
new_data = pd.concat([normal_sample, fraud], ignore_index= True)
print(new_data['Class'].value_counts())
print(new_data.head())
# UnderSampling the Dataset
# Split the dataset in train and test
print("--------------------------------------Split dataset into Target and ind----------------------------------------------")
X = new_data.iloc[:, :-1].values
print(X)
y = new_data.iloc[:, -1].values
print(y)
# ------------------------------------split data into training and test--------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
print("------------------------------------Training the Model--------------------------------------------")
print("------------------------------------Logistic Regression--------------------------------------------")
# Import Ml model to dataset
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

print("------------------------------------DecisionTreeClassifier--------------------------------------------")
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred1 = tree.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(recall_score(y_test, y_pred1))
print(precision_score(y_test, y_pred1))
print(f1_score(y_test, y_pred1))

print("------------------------------------RandomForestClassifier--------------------------------------------")
from sklearn.ensemble import RandomForestClassifier
rrc = RandomForestClassifier()
rrc.fit(X_train, y_train)
y_pred3 = rrc.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(recall_score(y_test, y_pred3))
print(precision_score(y_test, y_pred3))
print(f1_score(y_test, y_pred3))

print("------------------------------------Analyse Which model is best in undersampling--------------------------------------------")

obs = pd.DataFrame({'Model': ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"],
       'Acc': [accuracy_score(y_test, y_pred)*100, accuracy_score(y_test, y_pred1)*100, accuracy_score(y_test, y_pred3)*100]})
print(obs.head())
sns.barplot(x='Model', y='Acc', data= obs)
plt.show()


print("------------------------------------Over Sampling--------------------------------------------")
X = Dataset.iloc[:, :-1].values
print(X.shape)
y = Dataset.iloc[:, -1].values
print(y.shape)

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)
print(X_res.shape)
print(y_res.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size= 0.2, random_state= 42)

from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression()
lr1.fit(X_train, y_train)
y_pred = lr1.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

print("------------------------------------DecisionTreeClassifier--------------------------------------------")
from sklearn.tree import DecisionTreeClassifier
tree1 = DecisionTreeClassifier()
tree1.fit(X_train, y_train)
y_pred1 = tree1.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(recall_score(y_test, y_pred1))
print(precision_score(y_test, y_pred1))
print(f1_score(y_test, y_pred1))

print("------------------------------------RandomForestClassifier--------------------------------------------")
from sklearn.ensemble import RandomForestClassifier
rrc1 = RandomForestClassifier()
rrc1.fit(X_train, y_train)
y_pred3 = rrc1.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(recall_score(y_test, y_pred3))
print(precision_score(y_test, y_pred3))
print(f1_score(y_test, y_pred3))

print("------------------------------------Analyse Which model is best in Over Sampling--------------------------------------------")
obs1 = pd.DataFrame({'Model': ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"],
       'Acc': [accuracy_score(y_test, y_pred)*100, accuracy_score(y_test, y_pred1)*100, accuracy_score(y_test, y_pred3)*100]})

print(obs1.head())
sns.barplot(x='Model', y='Acc', data= obs1)
plt.show()


rf = RandomForestClassifier()
rf.fit(X_res, y_res)
import joblib

joblib.dump(rf, 'Credit_card_fraud_detect')
model = joblib.load('Credit_card_fraud_detect')
predd = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

if predd == 1:
  print("Fraud")
else:
  print("Normal")


