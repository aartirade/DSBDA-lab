#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd 
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:\Users\aarti\Downloads\titanic.csv")
df


# In[9]:


df.columns


# In[10]:


df.isnull().sum()


# In[11]:


Q1=df['Age'].quantile(0.25)
Q3=df['Age'].quantile(0.75)
IQR=Q3-Q1
print("IQR(", IQR, ") =", "Q3(", Q3, ")- Q1(", Q1, ")")


# In[12]:


lower_limit=Q1-IQR
upper_limit=Q3+IQR
lower_limit,upper_limit


# In[13]:


df_without_outliers=df[(df['Age']>lower_limit)&(df['Age']<upper_limit)]
df_without_outliers


# In[14]:


sns.boxplot(x='Age' , y='Sex', hue='Survived' ,data = df)


# In[15]:


sns.boxplot(x='Age' , y='Sex', hue='Survived' ,data = df_without_outliers)


# In[16]:


sns.violinplot(x='Sex', y='Age', data=df, hue="Survived",split=True)


# In[17]:


sns.displot(df['Age'])


# In[18]:


sns.catplot(x='Sex',y='Age' , data=df,hue='Survived')


# In[19]:


sns.swarmplot(x='Sex', y='Age', data=df, hue='Survived')


# In[20]:


sns.stripplot(x='Sex', y='Age', data=df, hue='Survived')


# In[ ]:




