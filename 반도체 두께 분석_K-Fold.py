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


# # PCA 진행
# pca = PCA(n_components=27)
# pca.fit(train_X)
#
# train_score = pca.transform(train_X)
# test_score = pca.transform(test_X)
#
# pd.DataFrame(pca.explained_variance_ratio_).plot(kind='bar', figsize=(20,5))
#
# for i in [10, 27, 50, 100, 120, 150, 226]:
#     print(i, " : ", pca.explained_variance_ratio_[:i].sum())
#
# PCA_NUM = [10, 27, 100, 226]
# history_save = []
# model_save = []
#
# for i, n in enumerate(PCA_NUM):
#     print(f"{i+1}번 모델")
#     xData = train_score[:,:n]
#     xTestData = test_score[:,:n]
#     print(xData.shape)


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
    model.add(Dense(units=4, activation='linear'))

    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    return model

# split 개수, 셔플 여부 및 seed 설정
kf = KFold(n_splits = 5, shuffle = True, random_state = 32)

import gc

pred_list = []

epochs = 300
batch_size = 4000
validation_split = 0.05
for train_index, valid_index in kf.split(train_X):
    X_train, X_valid = train_X.iloc[train_index], train_X.iloc[valid_index]
    y_train, y_valid = train_Y.iloc[train_index], train_Y.iloc[valid_index]

    model = createmodel()

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid))

    pred_test = model.predict(test_X)

    # 변수 삭제
    del model

    # 메모리 해제
    gc.collect()

    pred_list.append(pred_test)


pred_list_mean = np.array(pred_list).mean(axis=0)


sample_sub = pd.read_csv('dataset/sample_submission.csv', index_col=0)
submission = sample_sub+pred_list_mean
submission.to_csv('submission_kfold.csv')


