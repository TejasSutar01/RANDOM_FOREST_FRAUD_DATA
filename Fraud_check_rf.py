# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:27:27 2020

@author: tejas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
fraud=pd.read_csv("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\RANDOM FORESTS\FRAUD CHECK\Fraud_check.csv")
fraud.isnull().sum()
fraud.rename(columns={"Undergrad":"UG","Marital.Status":"Marital","City.Population":"Population","Work.Experience":"exp"},inplace=True)
fraud["income"]="<=35000"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"

#####Label Encoder#############
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
select_columns=["UG","Marital","Urban","income"]
le.fit(fraud[select_columns].values.flatten())
fraud[select_columns]=fraud[select_columns].apply(le.fit_transform)
fraud=fraud.drop(["Taxable.Income"],axis=1)
features=fraud.iloc[:,0:5]
labels=fraud.iloc[:,5]

#####Train n Test#########
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier as rf                     
model=rf(n_jobs=4,n_estimators=150,oob_score=True,criterion="entropy")                     
model=model.fit(x_train,y_train)
model.oob_score

####train prediction########
train_pred=model.predict(x_train)
####train confusion matrix###########
train_con=confusion_matrix(y_train,train_pred)
####train accuracy####
from sklearn.metrics import accuracy_score
train_accu=accuracy_score(y_train,train_pred)######100%

########test_pred##########
test_pred=model.predict(x_test)
#####confusion matrix test#########
test_con=confusion_matrix(y_test,test_pred)
####accuracy#########
test_accu=accuracy_score(y_test,test_pred)#######72%

###graph##########
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
colnames=list(fraud.columns)
predictors=colnames[:5]
target=colnames[5]

tree = model.estimators_[5]

dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

## Creating pdf and png file the selected decision tree
graph.write_pdf('fraud_rf.pdf')
graph.write_png('fraud_rf.png')
pwd
