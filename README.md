# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DELLI PRIYA L
RegisterNumber: 212222230029
*/

import pandas as pd
data=pd.read_csv('/content/Employee.csv')

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### Data Head:
![image](https://github.com/Priya-Loganathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121166075/eb6f8b57-3c5f-41bd-9221-f77c4c997b9e)
### Dataset Info:
![image](https://github.com/Priya-Loganathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121166075/73090ea7-f064-4917-b035-b6f529bb5f34)
### Null Dataset:
![image](https://github.com/Priya-Loganathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121166075/e5c08043-7e42-457e-9fde-8590fa82053e)
###  Values Count in Left Column:
![image](https://github.com/Priya-Loganathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121166075/dd732811-bcff-4f5c-a950-3cf453e6b953)
### Dataset transformed head
![image](https://github.com/Priya-Loganathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121166075/8e5e6127-f7bb-4ddf-9689-a20d23299bff)
### x.head:
![image](https://github.com/Priya-Loganathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121166075/465b1d28-f520-4ced-9455-0a591ef002b3)
### Accuracy:
![image](https://github.com/Priya-Loganathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121166075/f0077a3d-04f9-405a-9857-e7ce14e4c356)
### Data Prediction:
![image](https://github.com/Priya-Loganathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121166075/4f4f4796-9fbd-48f8-95b7-3805e110c1fd)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
