import pandas as pd
import numpy as np
import warnings
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.layers import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold


train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')



train_X = train.iloc[:,4:]
train_Y = train.iloc[:,0:4]
test_X = test.iloc[:,1:]


# PCA 진행
pca = PCA(n_components=13)
pca.fit(train_X)

pca_train_X = pca.transform(train_X)
pca_test_X = pca.transform(test_X)
print(type(pca_train_X))
pca_train_X = pd.DataFrame(pca_train_X)
pca_test_X = pd.DataFrame(pca_test_X)


# K-fold를 위한 모델함수 생성
def createmodel():
    model = Sequential()
    model.add(Dense(units=1000, activation='relu', input_dim=27))
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

    return model

# split 개수, 셔플 여부 및 seed 설정
kf = KFold(n_splits = 5, shuffle = True, random_state = 32)

import gc

pred_list = []

epochs = 10
batch_size = 500
for train_index, valid_index in kf.split(pca_train_X):
    X_train, X_valid = pca_train_X.iloc[train_index], pca_train_X.iloc[valid_index]
    y_train, y_valid = train_Y.iloc[train_index], train_Y.iloc[valid_index]

    model = createmodel()

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid))

    pred_test = model.predict(pca_test_X)

    # 변수 삭제
    del model

    # 메모리 해제
    gc.collect()

    pred_list.append(pred_test)


pred_list_mean = np.array(pred_list).mean(axis=0)

sample_sub = pd.read_csv('dataset/sample_submission.csv', index_col=0)
submission = sample_sub+pred_list_mean
submission.to_csv('submission_pca+kfold.csv')