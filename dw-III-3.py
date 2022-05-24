#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r'C:\Users\aarti\Downloads\archive (9)\Salary_Data.csv')
df.head(10)


# In[3]:


df.tail(5)


# In[4]:


df.describe()


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[8]:


df_salary = df.groupby(['YearsExperience'], as_index=False).agg(mean=('Salary', 'mean'),minimum=('Salary', 'min'), maximum=('Salary', 'max'),median=('Salary', 'median'),standard_deviation=('Salary', 'std'))


# In[9]:


df_salary 


# In[10]:


df_salary.plot('YearsExperience',['mean','median','standard_deviation'],figsize=(15, 8))


# In[12]:


#2. iris

import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\aarti\Downloads/Iris.csv')
df


# In[13]:


mean_df = df.groupby(['Species'], as_index=False).agg('mean')
mean_df


# In[14]:


mean_df[['SepalLengthCm','SepalWidthCm','PetalWidthCm','PetalLengthCm']].plot(figsize=(12,7))


# In[15]:


std_df = df.groupby(['Species'], as_index=False).agg('std')


# In[16]:


std_df.plot('Species',['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],figsize=(15, 8))


# In[ ]:




