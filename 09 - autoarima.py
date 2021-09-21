# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:50:07 2020

@author: marco
turotial original: https://scikit-learn.org/dev/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
mudanças no idex e na lib pyramid(agora é pmdarima)
não consegui usar a lib plotly 
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
#import matplotlib.pyplot as plt

#carregar dados e anexar datas

data = pd.read_csv('train.csv',index_col=0)
data.index = pd.to_datetime(data.index)
data.index = pd.date_range('2018-06-01', periods=744, freq='H')

#componentes
df = pd.Series(data['Total Power'])
result = seasonal_decompose(df, model='multiplicative')
fig1 = result.plot(0) #coloquei o 0 para plotar em figuras diferentes

#rodar o auto_arima (max_p=9/max_q=9/m=24 overfitting)
from pmdarima import auto_arima
stepwise_model = auto_arima(df, start_p=1, start_q=1,
                           max_p=5, max_q=5, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

#separar conjunto, treinar e plotar
#treinando 30 dias e prevendo o próximo
train = df.loc['2018-06-01':'2018-07-01']
test = df.loc['2018-07-01':]

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=24)
print(future_forecast)

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
fig1 = pd.concat([test,future_forecast],axis=1)
fig1.plot()


#métricas
from sklearn.metrics import mean_absolute_error as mae
MAE = mae(test,future_forecast)
print(MAE) 
# n= 12 .. 0.1648896586820001 / n=24 0.04181662074629939
from sklearn.metrics import mean_squared_error as mse
MSE = mse(test,future_forecast)
print(MSE) 
#n=12 ..0.042302582696258806 / n=24 0.0026565072413271226

#Best model:  ARIMA(4,1,3)(2,1,0)[24] 


