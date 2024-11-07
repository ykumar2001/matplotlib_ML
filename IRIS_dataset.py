import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


iris=load_iris()
print(iris.data)
print(iris.target)

x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train,y_train)

y_pred=KNN.predict(x_test)

print("predicted",y_pred)
print("actual",y_test)

accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

cm=confusion_matrix(y_test,y_pred)
print("confusion matrix:",cm)

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("logistic regression accuracy:",accuracy)
