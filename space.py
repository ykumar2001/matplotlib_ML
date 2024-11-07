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
predicted_value=lr.predict([[1,0,0,160000,130000,300000]])
print(predicted_value)

import matplotlib.pyplot as pt
pt.scatter(x_test[:, 3], y_test, color='red', label='Actual values')  # Plot actual points from test set
pt.scatter(160000, predicted_value, color='blue', label='Predicted value')  # Plot the predicted point

# Adding labels and title
pt.xlabel("R&D Spend")  # Change this according to the feature you want to plot (here it's the 4th feature)
pt.ylabel("Profit")
pt.title("Profit vs R&D Spend")

# Show legend
pt.legend()

# Show the plot
pt.show()