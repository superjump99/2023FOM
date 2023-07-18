import pandas as pd
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization


warnings.filterwarnings('ignore')
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')


#독립변수와 종속변수를 분리합니다.
train_X = train.iloc[:,4:]
train_Y = train.iloc[:,0:4]
test_X = test.iloc[:,1:]

#케라스를 통해 모델 생성을 시작합니다.


model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=226))
model.add(Dropout(0.2))
model.add(Dense(units=700, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation='linear'))

#모델을 컴파일합니다.
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

#모델을 학습합니다.

model.fit(train_X, train_Y, epochs=300, batch_size=4000, validation_split = 0.05)

#예측값을 생성합니다.
pred_test = model.predict(test_X)

#submission 파일을 생성합니다.
sample_sub = pd.read_csv('dataset/sample_submission.csv', index_col=0)
submission = sample_sub+pred_test
submission.to_csv('submission.csv')