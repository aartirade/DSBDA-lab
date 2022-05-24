#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


# In[7]:


df = pd.read_csv(r"C:\Users\aarti\Downloads\StudentsPerformance.csv")


# In[8]:


df.head(10)


# In[9]:


df.isnull().any()


# In[10]:


df.isnull().sum()


# In[11]:


df['math score'].fillna(value=df['math score'].mean(),inplace=True)
df['writing score'].fillna(value=df['writing score'].mean(),inplace=True)
df['reading score'].fillna(value=df['reading score'].mean(),inplace=True)


# In[67]:


import seaborn as sns
sns.boxplot(x=df['math score'])


# In[12]:


#outliers

outliers = []
def detect(df):
    threshold = 3
    mean = np.mean(df)
    std = np.std(df)
    
    for d in df:
        z_score = (d-mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(d)
    return outliers
    


# In[13]:


var='math score'
z_scores=detect(df[var])
outliers=df[df[var].isin(z_scores)]
outliers


# In[14]:


dfs = df[~df.index.isin(outliers.index)]
dfs


# In[15]:


dfs.skew(axis =0)


# In[16]:


dfs.hist(alpha=0.5, figsize=(16, 10))


# In[19]:


sns.kdeplot(dfs['writing score'])


# In[20]:


a = np.log(dfs['writing score'])

a.skew(axis=0)

sns.kdeplot(a)


# In[24]:


plt.figure(figsize=(14,8))
plt.subplot(1,2,1) ## means 1 row , 2 columns and 1st plot
dfs['writing score'].hist(bins=30)

plt.subplot(1,2,2)
stats.probplot(dfs['writing score'], dist="norm", plot=plt)
plt.show()


# In[ ]:




