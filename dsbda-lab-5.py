#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


sna_df = pd.read_csv(r'C:\Users\aarti\Downloads\Social_Network_Ads.csv')
sna_df


# In[3]:


sna_df.isnull().sum()


# In[4]:


sna_df.describe()


# In[5]:


sna_df = sna_df.replace('Male',1)


# In[6]:


sna_df = sna_df.replace('Female',0)
sna_df


# In[7]:


sna_df.corr()


# In[8]:


#Spliting the dataset in independent and dependent variables
X = sna_df.loc[:, ['Age', 'EstimatedSalary','Gender']].values
y = sna_df['Purchased'].values


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 45)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[10]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logisticregression = LogisticRegression()
logisticregression.fit(X_train, y_train)


# In[11]:


y_pred = logisticregression.predict(X_test)
print(y_pred)


# In[12]:


y_compare = np.vstack((y_test,y_pred)).T
print(y_compare)


# In[13]:


#2.  Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('true negatives (TN): Both, actual and predicted values are false: ', cm[0,0])
print('true positives (TP): Both, actual and predicted values are true: ', cm[1,1])
print('false positives (FP): Predicted value is yes but actual is false: ', cm[0,1])
print('false negative (FN): Predicted value is no but actual is true: ', cm[1,0])


# In[14]:


#accurecy 
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)*100
score


# In[15]:


#error rate
#Mean Squared error
print(np.mean((y_pred-y_test)**2))


# In[17]:


precision = cm[1,1]  / (cm[1,1] +  cm[0,1] ) 
precision


# In[18]:


recall = cm[1,1]  / (cm[1,1] + cm[1,0] )
recall


# In[ ]:




