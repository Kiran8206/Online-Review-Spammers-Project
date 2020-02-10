#!/usr/bin/env python
# coding: utf-8

# In[140]:


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[141]:


movie_review = pd.read_csv(r"C:\Users\yaram\Downloads\Movie_review.csv")


# In[142]:


df = pd.DataFrame(movie_review)
df.head()


# Punkt Sentence Tokenizer. This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used.

# In[143]:


#import nltk
#nltk.download('punkt')  #this is used to download punkt 


# In[144]:


from nltk import word_tokenize,sent_tokenize


# In[145]:


df["tokenized_text"] = df["text"].apply(word_tokenize)
#df["tokenized_text"]


# In[146]:


df.head()


# Lets remove stop which we don't require

# In[147]:


stop =['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


# In[148]:


def rmstopwords(y):
  A = []
  for i in y:
    if i not in stop:
      A.append(i)   
  return A


# In[149]:


lista=df['tokenized_text']
listb=[]
for item in lista:
    b = rmstopwords(item)
    listb.append(b)   
df['text_without_keywords']=listb
#df['text_without_keywords'].head()


# In[150]:


df.head()


# In orde to calicualte the sentiment score. AFINN is a list of words rated for valence with an integer between minus five (negative) and plus five (positive).

# In[151]:


afinn = pd.read_csv(r"C:\Users\yaram\Downloads\Afinn.csv",encoding = "ISO-8859-1")
df_afinn = pd.DataFrame(afinn)
df_afinn.head()


# Converting into dictionary for comparison

# In[152]:


afinn_dict = dict(zip(df_affin.Word,df_affin.Score))
#afinn_dict


# In[153]:


def sentiment_score(array):
  A = []
  for i in array:
    if i in afinn_dict:
      A.append(afinn_dict[i])
    else:
      A.append(0)
  return A


# In[154]:


lista = df["text_without_keywords"]
listb = []
listc = []
#print(lista)
for arrays in lista:
    listb = sentiment_score(arrays)
    listc.append(np.sum(listb))
df["sentiment_scores"] = listc


# In[155]:


df["sentiment_scores"].head()


# In[156]:


df["sentiment_scores"].describe()


# In[ ]:




