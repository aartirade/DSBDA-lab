#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import sklearn as sk
import math
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


# In[2]:


#sentence token
from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
# nltk.download('punkt')
tokenized_text=sent_tokenize(text)
print(tokenized_text)


# In[3]:


#word token
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)


# In[4]:


#Frequency Distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)


# In[5]:


fdist.most_common(2)


# In[6]:


import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()


# In[7]:


#stopwords
from nltk.corpus import stopwords
# nltk.download('stopwords')
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[8]:


#removing stopwords
filtered_sent=[]
for w in tokenized_text:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_text)
print("Filterd Sentence:",filtered_sent)


# In[9]:


#stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


# In[10]:


#Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))


# In[11]:


#POS tagging
sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(sent)
print(tokens)


# In[13]:


nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(tokens)


# In[ ]:




