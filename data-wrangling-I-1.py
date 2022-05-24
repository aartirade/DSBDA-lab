#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np


# In[9]:


#loading dataset
df = pd.read_csv(r'C:\Users\aarti\Downloads\abalone.csv')  


# In[10]:


print(df)


# In[11]:


#data preprocessing
df.head()   


# In[12]:


df.tail()


# In[13]:


df.describe()


# In[14]:


#checking dimentions
df.shape


# In[15]:


df.size


# In[16]:


df.ndim


# In[17]:


#checking for missing values 
df.isnull()


# In[18]:


df.notnull()


# In[19]:


df.isnull().sum()


# In[21]:


#data formatting
#coverting float to int type
df['Height'] = df['Height'].astype(int)
display(df.dtypes)


# In[22]:


from sklearn import preprocessing


# In[23]:


#normalize data 

from sklearn import preprocessing
x_array = np.array(df['Rings'])
normalise = preprocessing.normalize([x_array])
print(normalise)


# In[24]:


pd.get_dummies(df['Rings'])    #onverting categorical data into dummy 


# In[25]:


df


# In[31]:


#converting categorical into quantitative variable
conv= df['Sex'].replace(['M', 'F'], 
                        [0, 1], inplace=False)


# In[32]:


print(conv)


# In[ ]:




