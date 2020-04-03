#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:18:32 2020

@author: bhavin
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/bhavin/Downloads/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)
cm


from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

binary = np.array([[55, 42],
                   [12, 91]])

fig, ax = plot_confusion_matrix(conf_mat=binary)
plt.show()



from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred1)
acc

#KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred2)
cm1
from sklearn.metrics import accuracy_score
acc1=accuracy_score(y_test,y_pred2)

acc1

binary = np.array([[74, 23],
                   [55, 48]])

fig, ax = plot_confusion_matrix(conf_mat=binary)
plt.show()

#Decision tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3= confusion_matrix(y_test, y_pred3)
cm3
acc2=accuracy_score(y_test,y_pred3)
acc2

binary = np.array([[74, 23],
                   [35, 68]])

fig, ax = plot_confusion_matrix(conf_mat=binary)
plt.show()

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred4= classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4= confusion_matrix(y_test, y_pred4)
cm4


binary = np.array([[87, 10],
                   [46, 57]])

fig, ax = plot_confusion_matrix(conf_mat=binary)
plt.show()

acc3=accuracy_score(y_test,y_pred4)
acc3

#SVC
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred5 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test, y_pred5)

cm6

binary = np.array([[74, 23],
                   [33, 70]])

fig, ax = plot_confusion_matrix(conf_mat=binary)
plt.show()

acc4=accuracy_score(y_test,y_pred5)
acc4

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred6 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test, y_pred6)
cm6
acc6=accuracy_score(y_test,y_pred6)
acc6

binary = np.array([[76, 21],
                   [37, 66]])

fig, ax = plot_confusion_matrix(conf_mat=binary)
plt.show()


#ROC Curve for naive bayes
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_test,logmodel.predict_prob(X_test)[:,1])

