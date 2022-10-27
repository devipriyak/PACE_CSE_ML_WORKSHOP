import pandas as pd
import numpy as np
import tkinter
from tkinter import *
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
def polyregr():
    data = pd.read_csv('E:/attendance.csv',nrows=1000)
    data['Attendace']=data['Attendace'].fillna(value=data['Attendace'].mean())
    data['marks']=data['marks'].fillna(value=data['marks'].mean())

    x=data[['Attendace']]
    y=data['marks']
    pf1=PolynomialFeatures(degree=4)
    x1=pf1.fit_transform(x)
    regr=LinearRegression()
    regr.fit(x1,y)
    v=entry.get()
    pred=np.array([[v]], dtype=float)
    p=regr.predict(pf1.fit_transform(pred))
    t1.delete(1.0,END)
    t1.insert(END,p[0])

#gui
root =Tk()
#frame size
root.geometry("1000x200")
root.configure(background='black')

#label for "enter the Salinity level"

NameLb = Label(root, text="ENTER Attendance:", fg="White",bg="Black")
NameLb.config(font=("Times",20,"bold"))
NameLb.grid(row=6, column=1, pady=20, sticky=W)

#input entry text box
entry= Entry(root,width=40)
entry.grid(row=6,column=2)
#t1 = Text(root, height=1, width=40,bg="Black")
#t1.config(font=("arial",15,"bold italic"))
#t1.grid(row=6, column=2, padx=10)

#prediction button
dst = Button(root, text="PREDICT", command=polyregr,fg="Red",bg="Black")
dst.config(font=("Times",15,"bold"))
dst.grid(row=12, column=2,padx=10)


#label for "enter the predictded insulin"
NameLb = Label(root, text="THE PREDICTED Marks are:", fg="White",bg="Black")
NameLb.config(font=("Times",20,"bold"))
NameLb.grid(row=10, column=1, pady=20, sticky=W)

t1 = Text(root, height=1, width=40,bg="Black",fg="White")
t1.config(font=("arial",15,"bold"))
t1.grid(row=10, column=2, padx=10)
root.mainloop()