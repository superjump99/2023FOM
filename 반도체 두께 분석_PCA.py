import pandas as pd
import warnings
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.layers import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')


train_X = train.iloc[:,4:]
train_Y = train.iloc[:,0:4]
test_X = test.iloc[:,1:]


# PCA 진행
pca = PCA(n_components=13)
pca.fit(train_X)

train_score = pca.transform(train_X)
test_score = pca.transform(test_X)

pd.DataFrame(pca.explained_variance_ratio_).plot(kind='bar', figsize=(20,5))

# for i in [10, 27, 50, 100, 120, 150, 226]:
#     print(i, " : ", pca.explained_variance_ratio_[:i].sum())

PCA_NUM = [10, 13]
history_save = []
model_save = []

for i, n in enumerate(PCA_NUM):
    print(f"{i+1}번 모델")
    xData = train_score[:,:n]
    xTestData = test_score[:,:n]
    print(xData.shape)

model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=13))
model.add(Dropout(0.2))
model.add(Dense(units=700, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation='linear'))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])

epochs = 10
batch_size = 500
validation_split = 0.05
history = model.fit(xData, train_Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

pred_test = model.predict(xTestData)

history_save.append(history)
model_save.append(model)


sample_sub = pd.read_csv('dataset/sample_submission.csv', index_col=0)
submission = sample_sub+pred_test
submission.to_csv('submission_pca.csv')
