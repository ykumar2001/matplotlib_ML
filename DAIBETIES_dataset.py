import pandas as pd
from sklearn.datasets import load_diabetes

db=load_diabetes()
df=pd.DataFrame(data=db['data'],columns=db['feature_names'])
df['target']=db['target']
print(df.head())
# print(df.describe())
# print(df.isnull().sum())


from sklearn.model_selection import train_test_split
x=db['data']
y=db['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("length of train:",len(x_train))
print("test set size:",len(x_test))

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred)
print(f"Mean squared error:{mse:.2f}")
r2=r2_score(y_test,y_pred)
print(f"R-squared:{r2:.2f}")

import matplotlib.pyplot as pt 
pt.xlabel('actual')
pt.ylabel('predicted')
pt.title("actual vs predicted")
pt.show()


from sklearn.tree import DecisionTreeClassifier
# dt=DecisionTreeClassifier(random_state=42)
# dt.fit(x_train,y_train)
# y_pred=dt.predict(x_test)
# print("decision the acccuracy:",mean_squared_error(y_test,y_pred))

from sklearn.ensemble import RandomForestRegressor
# rf=RandomForestRegressor(random_state=42)
# rf.fit(x_train,y_train)
# y_pred=rf.predict(x_test)
# print("random forest mse:",mean_squared_error(y_test,y_pred))

from sklearn.svm import SVC
# svm=SVC(kernel='linear')
# svm.fit(x_train,y_train)
# y_pred=svm.predict(x_test)
# print("svm accuracy:",mean_squared_error(y_test,y_pred))

