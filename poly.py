import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv('attendance.csv')
data[' Attendance ']=data[' Attendance '].fillna(value=data['Attendance'].mean())
data['marks']=data['marks'].fillna(value=data['marks'].mean())
x=data[['Attendance']]
y=data['marks']
pf1=PolynomialFeatures(degree=4)
x1=pf1.fit_transform(x)  #covert original x into polynomial features
regr=LinearRegression()
regr.fit(x1,y)
y_pred=regr.predict(x1)
R_square = r2_score(y,y_pred)
print('Coefficient of Determination:', R_square) 
ch='y'
while(ch=='y' or ch=='Y'):
    sal=float(input("Enter atten to Predict :"))
    sal1=pf1.fit_transform([[sal]])
    p=regr.predict(sal1)
    print("\nTemperature is ",p)
    ch=input("Enter y to calculate more : ")
