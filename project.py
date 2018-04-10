# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:31:15 2018

@author: adity
"""

import numpy as np
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import seaborn as sns

file =pd.read_csv('Desktop/datafile.csv')
file.head()

file.info()

file.describe()
file.hist(figsize=(15,15))
file.columns
#sort = file.head()
#sort
#plt.plot(sort)
#plt.plot(x= 'YEAR')
#
#plt.show()

fig = plt.figure(figsize=(6,6))
ax= fig.add_subplot(111)
dfg = file.groupby('YEAR').sum()['ANN']
dfg.plot('line', title ="Rainfall in each year in india", fontsize =20)
plt.ylabel('Rainfall in mm')
ax.title.set_size(20)
ax.xaxis.label.set_fontsize(10)
ax.yaxis.label.set_fintsize(10)

file.dropna(inplace=True)
file.isnull().sum()
file.replace([np.inf,-np.inf],np.nan,inplace=True)
file.head()

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
file.head(0)
features = []
for element in file.head(0):
    features.append(element)

features.remove('JAN')
features.remove('FEB')
features.remove('MAR')
features.remove('APR')
features.remove('OCT')
features.remove('NOV')
features.remove('DEC')
features.remove('AUG')
features.remove('SEP')
features.remove('Jan-Feb')
features.remove('Jun-Sep')
features.remove('Mar-May')
features.remove('Oct-Dec')
features.remove('ANN')
print(features)


feature2 = ['AUG']
print(feature2)
x  = file.loc[:, features].values
print(x)
# Separating out the target
y = file.loc[:,feature2].values
print(y)
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)



##from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#le.fit(X_train)
#list(le.classes_)
#file.apply(LabelEncoder().fit_transform)




#class MultiColumnLabelEncoder:
#    def __init__(self,columns = None):
#        self.columns = columns # array of column names to encode
#
#    def fit(self,p,y=None):
#        return self # not relevant here
#
#    def transform(self,p):
#        
#      
#      #  Transforms columns of X specified in self.columns using
#       # LabelEncoder(). If no columns specified, transforms all
#        #columns in X.
#        
#        output = p.copy()
#        if self.columns is not None:
#            for col in self.columns:
#                output[col] = LabelEncoder().fit_transform(output[col])
#        else:
#            for colname,col in output.iteritems():
#                output[colname] = LabelEncoder().fit_transform(col)
#        return output
#
#    def fit_transform(self,p,y=None):
#        return self.fit(p,y).transform(p)
#
#MultiColumnLabelEncoder(columns = ['YEAR', 'MAY', 'JUN', 'JUL']).fit_transform(p)
#
#
#le = preprocessing.LabelEncoder()
#le.fit(p)

from sklearn.decomposition import PCA 
pca = PCA(n_components=4)
pca.fit(x)
print(pca.explained_variance_ratio_)


pca.score(x,y)

from sklearn import svm
from sklearn.datasets import make_classification
clf1 = svm.SVC()
X_train, y_train = make_classification()
X_test, y_test = make_classification()
clf1.fit(X_train, y_train)

clf1.score(X_test, y_test, sample_weight=None)



from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(max_depth=2, random_state=0)
clf2.fit(X_train, y_train)
clf2.predict(X_test)
clf2.score(X_test, y_test)

from sklearn.naive_bayes import GaussianNB
clf3 = GaussianNB()
clf3.fit(X_train, y_train)
clf3.predict(X_test)
clf3.score(X_test, y_test)

from sklearn.neural_network import MLPClassifier
clf4 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf4.fit(X_train, y_train)
clf4.predict(X_test)
clf4.score(X_test, y_test)

from sklearn import linear_model
clf5 = linear_model.LogisticRegression()
clf5.fit(X_train,y_train)
clf5.predict(X_test)
clf5.score(X_test,y_test)


plt.plot(clf1,clf2,clf3,clf4,clf5)
plt.show()
















































