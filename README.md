# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Suchitra Nath
RegisterNumber:  212223220112


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("df.head\n",df.head())
print("df.tail\n",df.tail())
x = df.iloc[:,:-1].values
print("Array value of x:",x)
y = df.iloc[:,1].values
print("Array value of y:",y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("Values of y predict:\n",y_pred)
print("Array values of y test:\n",y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## df.head
![image](https://github.com/user-attachments/assets/dc344a5a-cedd-4007-be03-9fcb81134fe7)

## df.tail
![image](https://github.com/user-attachments/assets/1a897782-448a-4ff1-8217-000b2b5ef6db)

## Array value of X
![image](https://github.com/user-attachments/assets/033fad71-bfa7-4506-b3ff-746563894a0a)

## Array value of Y
![image](https://github.com/user-attachments/assets/3ab1caf4-b899-4d43-9386-e465bffef739)

## Values of Y prediction
![image](https://github.com/user-attachments/assets/654bf50e-8be0-4215-b333-19f703613c93)

## Array values of Y test
![image](https://github.com/user-attachments/assets/066572fd-90e8-4879-b0f0-6dc3a01092a5)

## Training Set Graph
![image](https://github.com/user-attachments/assets/1454169e-4d85-49f3-8697-7ee4c801de7b)

## Test set graph
![image](https://github.com/user-attachments/assets/be1543ca-4f1a-4584-8d5b-5c671ca11aae)

## Values of MSE, MAE and RMSE
![image](https://github.com/user-attachments/assets/8f2f8312-52a6-44a5-962c-c5cea9de2155)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
