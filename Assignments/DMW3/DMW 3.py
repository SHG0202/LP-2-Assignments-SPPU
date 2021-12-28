#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().system('pip3 install apyori')
pd.set_option('display.width',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_colwidth',None)


# In[42]:


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


# In[50]:


dataset.head()


# In[52]:


for k in transactions:
    print(k)
    print(" ")


# In[44]:


from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# In[45]:


results = list(rules)


# In[55]:


for RelationRecord in results:
    print('Set of items: ',RelationRecord[0])
    print('Support: ',RelationRecord[1])
    print('Confidence:',RelationRecord[2][0][2],'for item1: ',RelationRecord[2][0][0],' and item2:',RelationRecord[2][0][1])
    print('Lift: ',RelationRecord[2][0][3])
    print('')


# In[39]:


def covert_to_list_for_df(results):
    item1=[tuple(i[2][0][0]) for i in results]
    item2=[tuple(i[2][0][1])[0] for i in results]
    support=[i[1] for i in results]
    confidence=[i[2][0][2] for i in results]
    lift=[i[2][0][3] for i in results]
    return list(zip(item1,item2,support,confidence,lift))
    
final_df=pd.DataFrame(covert_to_list_for_df(results),columns=['item1','item2','support','confidence','lift'])


# In[63]:


final_df


# In[ ]:




