#  Make a single prediction for  the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
# with the use of multiple linear regression in 50_Startups.csv

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('50_startups.csv')
x= df.iloc[:,:-1].values
y= df.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

# y_pred=lr.predict(x_test)

print(lr.predict([[1,0,0,160000,130000,300000]]))