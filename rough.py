import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()
df=pd.DataFrame(data=iris['data'],columns=iris['feature_names'])
df['target']=iris['target']
print(df.head())

import matplotlib.pyplot as pt
print(df.describe())
pt.show()

x=iris['data']
y=iris['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("training set size:",len(x_train))
print('test set size:',len(x_test))

model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("model accuracy:",accuracy)
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix:\n",cm)
pt.title("confusion matrix")
pt.xlabel('predicted')
pt.ylabel('true')
pt.show() 