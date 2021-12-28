#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#stopwords are useless words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
set(stopwords.words('english'))


# In[3]:


nltk.download('punkt')
from nltk.tokenize import word_tokenize

example_sent = "Hello. This is Mihir Kumar. First example of stop word filtration"

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

#filtered_sentence = []

#for w in word_tokens:
#    if w not in stop_words:
#        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)


# In[4]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

ps = PorterStemmer()

for w in filtered_sentence:
    print(ps.stem(w))


# In[5]:


data=pd.read_csv('IMDB Dataset.csv')


# In[6]:


data.head()


# In[7]:


data.count()


# In[8]:


import re
corpus = []
for i in range(0, 1500):
    review = re.sub('[^a-zA-Z]', ' ', data['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(max_features = 1500)
X = vector.fit_transform(corpus).toarray()
y = data.iloc[0:1500, 1].values


# In[13]:


print(X)
X.shape


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[15]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[16]:


y_pred = classifier.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[18]:


#from nltk.metrics.scores import precision, recall
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)


# In[19]:


print(f'Precision of the model = {precision}')
print(f'Recall of the model = {recall}')


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




