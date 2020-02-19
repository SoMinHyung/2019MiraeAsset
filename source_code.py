import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def last_ratio(today_v, last_v):
    return (today_v- last_v) / last_v

def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (prep_data['volume'].rolling(window).mean())
        prep_data['f_marketcap_ma{}'.format(window)] = prep_data['f_marketcap'].rolling(window).mean()
    return prep_data

def build_training_data(prep_data):
    training_data = prep_data

    #이평선 대비 변동치
    windows = [5, 10, 20]
    for window in windows:
        training_data['close_ma%d_ratio' % window] = last_ratio(training_data['close'], training_data['close_ma%d' % window])
        training_data['volume_ma%d_ratio' % window] = last_ratio(training_data['volume'], training_data['volume_ma%d' % window])
        training_data['f_marketcap_ma%d_ratio' % window] = last_ratio(training_data['f_marketcap'], training_data['f_marketcap_ma%d' % window])

    #전일대비 변동치
    training_data['USD_ratio'] = last_ratio(training_data['USD'],training_data['USD_last'])
    training_data['EUR_ratio'] = last_ratio(training_data['EUR'], training_data['EUR_last'])
    training_data['GBP_ratio'] = last_ratio(training_data['GBP'], training_data['GBP_last'])
    training_data['close_ratio'] = last_ratio(training_data['close'], training_data['last'])
    training_data['volume_ratio'] = last_ratio(training_data['volume'], training_data['volume_last'])
    training_data['f_marketcap_ratio'] = last_ratio(training_data['f_marketcap'], training_data['f_marketcap_last'])

    #전체 시총대비 외국인 투자비율
    training_data['fk_marketcap_ratio'] = training_data['f_marketcap'] / training_data['marketcap']

    return training_data


dataset_train = pd.read_csv("KOS.csv", encoding='UTF-8')


prep_data = preprocess(dataset_train)
dataset_train = build_training_data(prep_data)
training_set = dataset_train.dropna()


#환율 전일 대비 변동
#5, 10, 20일 종가 / 거래대금(거래량X) / 외국인의 시총의 변동
#종가 / 거래대금 / 외국인 시총의 전일대비변동 : close_ratio, volume_ratio, f_marketcap_ratio
#외국인의 kospi시총 차지비율 : fk_marketcap_ratio
features_training_data = ['date', 'Y', 'USD_ratio', 'EUR_ratio', 'GBP_ratio',
                          'close_ma5_ratio', 'volume_ma5_ratio', 'f_marketcap_ma5_ratio',
                          'close_ma10_ratio', 'volume_ma10_ratio', 'f_marketcap_ma10_ratio',
                          'close_ma20_ratio', 'volume_ma20_ratio', 'f_marketcap_ma20_ratio',
                          'close_ratio', 'volume_ratio','f_marketcap_ratio', 'fk_marketcap_ratio']
features_chart_data = ['date', 'Y', 'close', 'open', 'high', 'low', 'volume']


train_data = training_set[features_training_data]
chart_data = training_set[features_chart_data]

train_data['date'] = pd.to_datetime(train_data.date)
train_data['date'] = train_data.date.dt.strftime('%Y')+train_data.date.dt.strftime('%m')+train_data.date.dt.strftime('%d')
train_data['date'] = pd.to_numeric(train_data['date'], errors='coerce')

#데이터셋 분류
train_dataset = train_data[(train_data['date'] >= 20070116) & (train_data['date'] <= 20161228)]
valid_dataset = train_data[(train_data['date'] >= 20170102) & (train_data['date'] <= 20171228)]
test_dataset = train_data[(train_data['date'] >= 20180102) & (train_data['date'] <= 20181228)]


#0~1로 스케일링
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_set_scaled = sc.fit_transform(train_dataset)
valid_set_scaled = sc.fit_transform(valid_dataset)
test_set_scaled = sc.fit_transform(test_dataset)


#numpy array형을 dataframe으로 변환
column_name = ['date', 'Y', 'USD_ratio', 'EUR_ratio', 'GBP_ratio','close_ma5_ratio', 'volume_ma5_ratio', 'f_marketcap_ma5_ratio',
               'close_ma10_ratio', 'volume_ma10_ratio', 'f_marketcap_ma10_ratio',
               'close_ma20_ratio', 'volume_ma20_ratio', 'f_marketcap_ma20_ratio',
               'close_ratio', 'volume_ratio','f_marketcap_ratio', 'fk_marketcap_ratio']
train_df = pd.DataFrame(train_set_scaled, columns=column_name)
valid_df = pd.DataFrame(valid_set_scaled, columns=column_name)
test_df = pd.DataFrame(test_set_scaled, columns=column_name)

x_train = train_df.drop('Y',axis=1)
y_train = train_df[['Y']]

x_valid = valid_df.drop('Y',axis=1)
y_valid = valid_df[['Y']]

x_test = test_df.drop('Y',axis=1)
y_test = test_df[['Y']]

x_train = x_train.values
y_train = y_train.values

x_valid = x_valid.values
y_valid = y_valid.values

x_test =  x_test.values
y_test = y_test.values


#3차원 변형
X_train_t = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
X_valid_t = x_valid.reshape(x_valid.shape[0],x_valid.shape[1],1)
X_test_t = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


#머신러닝 모델 설계
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization, MaxPooling1D, Conv1D, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# LSTM 신경망
model = Sequential() # Sequeatial Model

model.add(Conv1D(64, kernel_size=8, padding='same', input_shape=(17,1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))

model.add(Conv1D(64, kernel_size=8, padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))

model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))

model.add(LSTM(256, return_sequences= True, dropout=0.3))
model.add(BatchNormalization())
model.add(LSTM(256, activation = 'relu', dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer= Adam(lr=0.001), metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
history = model.fit(X_train_t, y_train, epochs=150, callbacks=[early_stop], validation_data=(X_valid_t,y_valid))

#테스트셋의 오차
y_vloss = history.history['val_loss']

#학습셋의 오차
y_loss = history.history['loss']

# 결과 그래프 생성
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label="Trainset_loss")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#결과 혼동행렬 생성
y_pred = model.predict(X_test_t)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)
print((cm[0,0]+cm[1,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]))


from keras.models import load_model
model.save('result_model.h5')