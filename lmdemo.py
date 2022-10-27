import pandas as pd
data=pd.read_csv('E:/attendance.csv')
print(data)
#data.describe()
#data['Attendace']=data['Attendace'].fillna(value=data['Attendace'].mean())
#Store Attendance from data(dataframe) in to X in 2D for LinearRegression
print(data)
X=data[['Attendace']]  #input
print(X.shape)
#Store Marks from data(dataframe) in to y in 1D for LinearRegression
y=data['marks'] #output
print(y.shape)
#Split the training set and test data set from the original X and y using train_test_split()
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3)
 #Bulid the Linear Regression Model
from sklearn.linear_model import LinearRegression
#Call the LinearRegression() to create instance of model
model=LinearRegression()
model.fit(X_train,y_train)
#Model Prediction on X_test data 
y_pred=model.predict(X_test)
print("m= ",model.coef_)
print("c=",model.intercept_)
#Test on New Instance
print(model.predict([[68]]))
from sklearn.metrics import r2_score,mean_squared_error
print("R2 score fit”,r2_score(y_test,y_pred))
print("Mean squared error: “, mean_squared_error(y_test, y_pred))
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
