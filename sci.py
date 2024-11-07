# import libraries

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#load or create a data set 
data={'area':[2600,3000,3200,3600,4000],
      'price':[550000,565000,610000,680000,725000]}
df=pd.DataFrame(data)
x=df[['area']]
y=df[['price']]

# split into train,test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# choose train model
model= LinearRegression()
model.fit(x_train,y_train)

# prediction
y_pred=model.predict(x_test)
print("predicted values:",y_pred)
print("actual values:",y_test.values)

# evaluation
mse=mean_squared_error(y_test,y_pred)
r_squared=model.score(x_test,y_test)

print(f"mean squared error: {mse}")
print(f"r_squared: {r_squared}")