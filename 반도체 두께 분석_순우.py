#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization


# In[7]:


import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.layers import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')


# In[10]:


train_X = train.iloc[:,4:]
train_Y = train.iloc[:,0:4]
test_X = test.iloc[:,1:]


# In[11]:


pca = PCA(n_components=27)
pca.fit(train_X)


# In[12]:


train_score = pca.transform(train_X)
test_score = pca.transform(test_X)


# In[26]:


print(train_X.shape, test_X.shape)
print(train_score.shape, test_score.shape)


# In[14]:


pd.DataFrame(pca.explained_variance_ratio_).plot(kind='bar', figsize=(20,5))


# In[15]:


for i in [10, 27, 50, 100, 120, 150, 226]:
    print(i, " : ", pca.explained_variance_ratio_[:i].sum())


# In[17]:


PCA_NUM = [10, 27, 100, 226]
history_save = []
model_save = []


# In[18]:


for i, n in enumerate(PCA_NUM):
    print(f"{i+1}번 모델")
    xData = train_score[:,:n]
    xTestData = test_score[:,:n]
    print(xData.shape)

# In[23]:


model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=27))
model.add(Dropout(0.2))
model.add(Dense(units=700, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation='linear'))


# In[24]:


model.compile(loss='mae', optimizer='adam', metrics=['mae'])


# In[25]:


epochs = 300
batch_size = 4000
validation_split = 0.05
history = model.fit(xData, train_Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)


# In[ ]:


pred_test = model.predict(test_X)


# In[ ]:


history_save.append(history)
model_save.append(model)


# In[ ]:


sample_sub = pd.read_csv('sample_submission.csv', index_col=0)
submission = sample_sub+model_save
submission.to_csv('submission_sw.csv')


# In[ ]:




