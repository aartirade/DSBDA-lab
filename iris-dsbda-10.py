#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\aarti\Downloads/Iris.csv',header=None)
df.columns = ["col1","col2","col3","col4","col5","col6"]
df.head()


# In[5]:


df.info()
np.unique(df["col6"])


# In[6]:


df.describe()


# In[7]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# In[8]:


fig, axes = plt.subplots(2, 2, figsize=(16, 8))


axes[0,0].set_title("Distribution of First Column")
axes[0,0].hist(df["col1"]);

axes[0,1].set_title("Distribution of Second Column")
axes[0,1].hist(df["col2"]);

axes[1,0].set_title("Distribution of Third Column")
axes[1,0].hist(df["col3"]);

axes[1,1].set_title("Distribution of Fourth Column")
axes[1,1].hist(df["col4"]);


# In[24]:


df.boxplot()


# In[ ]:




