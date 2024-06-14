
import pandas as pd
import os
from sklearn.model_selection import train_test_split,GridSearchCV
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_squared_error


import joblib

dataLoc=os.path.join(os.path.dirname(__file__),"diabetes.xls")
df=pd.read_excel(dataLoc)
df.head()

"""### **Some Processing to understand the data**"""

count=0
for i in df.columns:
  count=0
  for j in df[i].isna():
    if j:
      count+=1
  print(i,count,max(df[i]),min(df[i]))

print(sum(1 for i in df["Outcome"] if i==1))

diabetes_data_y=df['Outcome']
diabetes_data_x=df.drop('Outcome',axis=1)

#standadizing the data
scaler=StandardScaler()
standardized_data_x=scaler.fit_transform(diabetes_data_x)

x_train,x_test,y_train,y_test=train_test_split(standardized_data_x,diabetes_data_y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)

"""### **Function to calculate the accuracy of different model**"""

def calculate_correctness_of_model(y_test,y_pred,model,param):
  miss_prediction=0
  for i,z in zip(y_test,y_pred):
    if i!=z:
      miss_prediction+=1
  print(model.__class__.__name__)
  print(miss_prediction,mean_squared_error(y_test,y_pred),accuracy_score(y_test,y_pred),"with the param",param)

"""

> Different model and different params are used in GridSearchCV to find the best model with its best param

"""

models=[LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),SVC()]
param_grids=[{ # Logistic Regression
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    },
    { # Decision Tree Classifier
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    { # Random Forest Classifier
        'n_estimators': [100],
        'max_depth': [10, 15],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    },
    { # Support Vector Classifier
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },]
for model,param_grid in zip(models,param_grids):
  grid_search=GridSearchCV(model,param_grid,cv=5)
  grid_search.fit(x_train,y_train)

  best_model=grid_search.best_estimator_
  best_param=grid_search.best_params_
  y_pred=best_model.predict(x_test)

  calculate_correctness_of_model(y_test,y_pred,best_model,best_param)

"""

> Among the different models used DecisionTreeClassifier model gave the best results so I have chosen it for the prediction
"""

best_model=DecisionTreeClassifier(max_depth=5,min_samples_leaf=2,min_samples_split=2)
best_model.fit(x_train,y_train)
y_pred=best_model.predict(x_test)

print("Accuracy using Decision Tree Classifier is ",accuracy_score(y_test,y_pred))

"""### **Saving the model to use it in backend**"""

joblib.dump(best_model,os.path.join("drive","MyDrive","diabetes.pkl"))

"""### **Testing**"""

saved_model=joblib.load(os.path.join("drive","MyDrive","diabetes.pkl"))
result=saved_model.predict([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
print(result)
result=saved_model.predict([[ 0, -0.71653347, -0.57412775 , 0.7818138,   0.95685965,  0.25478047,
  -0.1264714,   10]])
print(result)