# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and read the csv file.
2. Encoding the data and Import Decision tree classifier.
3. Fit the data in the model.
4. Find the MSE , r2 and the Predicted.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SRI SAI PRIYA. S
RegisterNumber:  212222240103
```
```
import pandas as pd
df=pd.read_csv("CSVs/Salary.csv")
df.head(10)
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head(10)
x=df[['Position','Level']]
y=df['Salary']
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(Xtrain,Ytrain)
Ypred=dt.predict(Xtest)
from sklearn import metrics
mse=metrics.mean_squared_error(Ytest,Ypred)
mse
r2=metrics.r2_score(Ytest,Ypred)
r2
dt.predict([[5,6]])
```
## Output:

# df.head()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119475702/eb70ad62-a21c-4197-8332-8660b3307e7a)

# df.info()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119475702/d9af7d30-2afb-476b-940e-2f3797e51123)

# df.isnull().sum()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119475702/33109b6f-fa6e-435c-aaf0-0b5f8565b2d5)

# After Label Encoding

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119475702/b128aef5-52c5-4cae-8157-a5f52a025be1)

# mse

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119475702/08b6ef34-f878-4c5c-8cec-292797a40a92)

# r2

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119475702/c51d0939-da88-4395-8fbd-0e7879d6daf2)

# dt.predict

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119475702/e7cc52f4-a0f2-43f2-8353-6a441d3af1c8)

# plt.show()

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119475702/b331da30-8628-4881-8aea-e2b3f9cbbcf8)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
