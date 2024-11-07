import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix
# load your dataset 
wine =load_wine()
df=pd.DataFrame(data=wine['data'],columns=wine['feature_names'])
df['target']=wine['target']
print(df.head())

print(df.describe())
print(df['target'].value_counts())

import matplotlib.pyplot as pt
pt.show()

# train & test set
x=wine['data']
y=wine['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print('length of train :',len(x_train))
print('test set size:',len(x_test))

# creating model=logistic Regression 
from sklearn.linear_model import LogisticRegression
# model= LogisticRegression(max_iter=200)
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)

# classification by KNN
from sklearn.neighbors import KNeighborsClassifier
# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train,y_train)

# y_pred=knn.predict(x_test)
# print("KNN accuracy:",accuracy_score(y_test,y_pred))

# classification by Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
# dt=DecisionTreeClassifier(random_state=42)
# dt.fit(x_train,y_train)
# y_pred=dt.predict(x_test)
# print("decision tree accuracy:",accuracy_score(y_test,y_pred))

# classification by svm
from sklearn.svm import SVC
# svm=SVC(kernel='linear')
# svm.fit(x_train,y_train)
# y_pred=svm.predict(x_test)
# print('svm accuracy:',accuracy_score(y_test,y_pred))

# classification  by random forest classifier
from sklearn.ensemble import RandomForestClassifier
# rf=RandomForestClassifier(random_state=42)
# rf.fit(x_train,y_train)
# y_pred=rf.predict(x_test)
# print("random forest accuracy:",accuracy_score(y_test,y_pred))

# classification by gridsearchcv
from sklearn.model_selection import GridSearchCV
param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,5,10]
}
grid_search= GridSearchCV(estimator=RandomForestClassifier(random_state=42),param_grid=param_grid,cv=5)
grid_search.fit(x_train,y_train)
print("best parmeter found:",grid_search.best_params_)
best_model=grid_search.best_estimator_
y_pred=best_model.predict(x_test)
print("best model accuracy:",accuracy_score(y_test,y_pred))                          

# accuray $ confusion matrix
accuracy=accuracy_score(y_test,y_pred)
print(f"model accuracy:{accuracy * 100:.2f}%")
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix:\n",cm)

pt.xlabel("predicted")
pt.ylabel('actual')
pt.show()