# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:35:24 2020

@author: marco
"""


import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
import math
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('train.csv',index_col=0)
df.index = pd.to_datetime(df.index)
df.index = pd.date_range('2018-06-01', periods=744, freq='H')

#data.shape
#verificar se tem dados faltantes
#data.isnull().sum()
#Substituir os dados faltantes
#plotar heatmap
#sns.heatmap(data.isnull())

plt.subplots(figsize=(25,10))
plt.ylabel('Total Power')
plt.title('Time Series')
plt.plot('Total Power', data = df)

data = df.filter(['Total Power'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_x = []
train_y = []
train_data = scaled_data[0:int(training_data_len-24),:]

for i in range(24,len(train_data)):
    train_x.append(train_data[i-24:i, 0])
    train_y.append(train_data[i,0])
        
train_x, train_y = np.array(train_x), np.array(train_y)
train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))

model = Sequential()
model.add(LSTM(units = 96, return_sequences = True, input_shape = (train_x.shape[1],1)))
model.add(Dropout(0.2))
model.add(Dense(1))

#Gerar Fluxograma
#keras.utils.plot_model(model = model, to_file = 'AxisBankLSTM.png')

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(train_x,train_y,batch_size=23,epochs=125)

test_data = scaled_data[training_data_len-24:,:]
test_x = []
test_y = dataset[training_data_len: len(data), :] 
for i in range(24,len(test_data)):
    test_x.append(test_data[i-24:i,0])

    
test_x = np.array(test_x)
test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))

prediction = model.predict(test_x)
prediction = scaler.inverse_transform(prediction)

rmse = np.sqrt(np.mean((prediction-test_y)**2))

train = data[:training_data_len]
actual = data[training_data_len:]
actual['Predictions'] = prediction

plt.figure(figsize=(16,6))
plt.title('Model')
plt.plot('Total Power', data = df)
plt.plot(actual['Predictions'])
plt.show()

