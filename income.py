# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 03:23:45 2020

@author: Bhavin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:/Users/Bhavin/Downloads/income.csv",na_values=[ " ? " ])
data.info()
data.isnull().sum()
data.describe()
data.columns
sns.heatmap(data.corr())
data['JobType'].value_counts()
data['occupation'].value_counts()

data1=pd.read_csv("C:/Users/Bhavin/Downloads/income.csv", na_values=[" ?" ])
data1.isnull().sum()
missing=data1[data1.isnull().any(axis=1)]
data2=data1.dropna(axis=0)

data2.corr()
gender=pd.crosstab(index= data2["gender"],
                   columns='count',normalize=True)
print(gender)

sal_stat=pd.crosstab(index= data2['gender'],
                     columns=data2['SalStat'],
                     margins=True,
                     normalize='index')
sal_stat


sal=sns.countplot(data2['SalStat'])
sal

sns.distplot(data2['age'],bins=10,kde=False)

sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()

plt.bar(data2['age'],data2['SalStat'])


#Logistic Regression

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000': 0,' greater than 50,000': 1})
print(data2['SalStat'])

#data3=data2.iloc[:,12:13]
#data4=data3['SalStat'].map({' less than or equal to 50,000': 0,' greater than 50,000': 1})

new_data=pd.get_dummies(data2,drop_first=True)
column_list=list(new_data.columns)
features=list(set(column_list)-set(['SalStat']))
print(features)

X=new_data[features].values
y=new_data['SalStat'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
classifier.intercept_
classifier.coef_

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

print("misclassified samples= %d" %(y_test!=y_pred).sum())
 

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

print("misclassified samples= %d" %(y_test!=y_pred).sum())

