#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()


# In[2]:


data= pd.read_csv(r'C:\Users\aarti\Downloads\boston_housing.csv')
data


# In[3]:


data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data.head(10)


# In[4]:


boston.target.shape

data['Price'] = boston.target
data.head()


# In[6]:


data.info()


# In[8]:


x = boston.data
y = boston.target

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2,random_state = 0)
print("xtrain shape : ", xtrain.shape)

print("xtest shape  : ", xtest.shape)

print("ytrain shape : ", ytrain.shape)

print("ytest shape  : ", ytest.shape)


# In[9]:


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(xtrain, ytrain)
y_pred = regressor.predict(xtest)

plt.scatter(ytest, y_pred, c='green')
plt.xlabel("Price: in $1000's")
plt.ylabel("Predicted value")
plt.title("True value vs Predicted vale: Linear Regression")
plt.show()


# In[10]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest, y_pred)
print("Mean square error: ", mse)


# In[ ]:




