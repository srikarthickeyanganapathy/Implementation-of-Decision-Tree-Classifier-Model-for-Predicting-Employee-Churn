# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import standard libraries in python for finding Decision tree classsifier model for predicting employee churn.
2. Initialize and print the Data.head(),data.info(),data.isnull().sum()
3. Visualize data value count.
4. Import sklearn from LabelEncoder.
5. Split data into training and testing.
6. Calculate the accuracy, data prediction by importing the required modules from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SRI KARTHICKEYAN GANAPATHY
RegisterNumber: 212222240102

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![238831862-b1162149-bbea-43a7-96a9-354fd108a151](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393842/2106874b-eebd-438a-a3e1-46eed277e34d)

![238831958-bfe60847-ed9a-487c-a700-b43dca8659a5](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393842/71e876dc-c83d-461d-aeec-7e834dd60fa7)

![238832170-7f03738f-ea1f-4338-a98b-1ef247f52708](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393842/832b5cf8-f3a0-4345-9a43-29cee62df70f)

![238834981-a4e857d7-6fa3-4dc1-b615-ec4f8433c511](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393842/b8bc81d8-4095-4027-afb5-cf2424d3346d)

![238832413-ff85899e-0a75-4138-9b2a-d9abcdf20138](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393842/46fdc5ca-d3b3-46d1-9813-7261556a6457)

![238832633-6bf35c61-a89f-4af3-bdaf-840d8b320af5](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393842/f327843d-aa40-47f0-b5aa-202133c4195a)

![238832895-3010e840-d901-478a-ba93-82eba77e27e9](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393842/db4ebc71-95ab-4924-b58c-19e2c2038d3e)

![238833279-ebb42f60-6046-426c-9759-e629b2a68b2c](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393842/6f618b4d-a675-4f21-9bb8-28c23575a11a)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
