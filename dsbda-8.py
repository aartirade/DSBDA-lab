#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = sns.load_dataset('titanic')
dataset.head()


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

sns.distplot(dataset['fare'])

sns.distplot(dataset['fare'], kde=False)

sns.jointplot(x='age', y='fare', data=dataset)

dataset = dataset.dropna()


# In[17]:


sns.rugplot(dataset['fare'])

sns.barplot(x='sex', y='age', data=dataset)

sns.boxplot(x='sex', y='age', data=dataset)

sns.boxplot(x='sex', y='age', data=dataset, hue="survived")

sns.violinplot(x='sex', y='age', data=dataset)

sns.stripplot(x='sex', y='age', data=dataset)

sns.swarmplot(x='sex', y='age', data=dataset)

titanic_df = pd.read_csv(r"C:\Users\aarti\Downloads\train_data.csv")
test_df    = pd.read_csv(r"C:\Users\aarti\Downloads\test_data.csv")
titanic_df.info()
print("-----------------------------------------")
test_df.info()


# In[6]:


embark_dummies_titanic  = pd.get_dummies(dataset['embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)


# In[7]:


titanic_df = dataset.join(embark_dummies_titanic)


# In[9]:


test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)


# In[19]:


fare_not_survived = dataset["fare"][dataset["survived"] == 0]
fare_survived     = dataset["fare"][dataset["survived"] == 1]


# In[13]:


avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])


# In[15]:


titanic_df['fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))


# In[16]:


avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)


# In[ ]:




