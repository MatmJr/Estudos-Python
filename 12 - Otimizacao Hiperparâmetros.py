from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice,uniform,lognormal
from hyperas import optim
from keras import Sequential
from keras.layers import Dense,Dropout,LSTM,GRU,SimpleRNN
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from lib import split_sequence,divisao_dados_temporais #funções para dividir os dados
import json

# Usei o exemplo do site: https://github.com/maxpumperla/hyperas
def data():
    
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df = pd.concat([df_train, df_test])
    serie = df['Total Power']
    
    X, Y = split_sequence(serie.values, n_steps_in=1, n_steps_out=1)
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_norm, Y_norm = scaler.fit_transform(X), scaler.fit_transform(Y)
    
    X_train, y_train, X_test, y_test= divisao_dados_temporais(X_norm, Y_norm, perc_treino=0.6)
    
    return X_train,y_train,X_test,y_test

def create_model(X_train, y_train,X_test,y_test):
    
    option = {{choice([1,2,3,4])}} #sortear um dos métodos, no resultado final (0...3) 
    
    #LSTM
    if option == 1:
        
        n_features = X_train.shape[1]
        n_steps_lstm = {{choice([2,4,8,16,24,32])}}
        n_samples = int(X_train.shape[0]/n_steps_lstm)
        n_samples_total = n_samples*n_steps_lstm
        X_train_final = X_train[:n_samples_total,:].reshape((n_samples
                                                            ,n_steps_lstm,
                                                            n_features))
        y_train_final = np.array([y_train[i] for i in range(n_steps_lstm-1,len(y_train),n_steps_lstm)])
        
        model = Sequential()
        
        model.add(LSTM({{choice([37,16,96,22,95,35,21,59,12,88,86,30])}}, activation={{choice(['relu',
                                                            'tanh',
                                                            'sigmoid'])}}, 
                                                    input_shape=(n_steps_lstm,n_features)))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense(1))
        
    #RNN
    if option == 2:
            
        n_features = X_train.shape[1]
        n_steps_rnn = {{choice([2,4,8,16,24,32])}}
        n_samples = int(X_train.shape[0]/n_steps_rnn)
        n_samples_total = n_samples*n_steps_rnn
        X_train_final = X_train[:n_samples_total,:].reshape((n_samples
                                                            ,n_steps_rnn,
                                                            n_features))
        y_train_final = np.array([y_train[i] for i in range(n_steps_rnn-1,len(y_train),n_steps_rnn)])
    
        model = Sequential()
      
        model.add(SimpleRNN({{choice([37,16,96,22,95,35,21,59,12,88,86,30])}}, activation={{choice(['relu',
                                                            'tanh',
                                                            'sigmoid'])}}, 
                                                            input_shape=(n_steps_rnn,n_features)))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense(1))
    #GRU
    if option == 3:
        
        n_features = X_train.shape[1]
        n_steps_gru = {{choice([2,4,8,16,24,32])}}
        n_samples = int(X_train.shape[0]/n_steps_gru)
        n_samples_total = n_samples*n_steps_gru
        X_train_final = X_train[:n_samples_total,:].reshape((n_samples
                                                            ,n_steps_gru,
                                                            n_features))
        y_train_final = np.array([y_train[i] for i in range(n_steps_gru-1,len(y_train),n_steps_gru)])

        model = Sequential()
        model.add(GRU({{choice([37,16,96,22,95,35,21,59,12,88,86,30])}}, activation={{choice(['relu',
                                                            'tanh',
                                                            'sigmoid'])}}, 
                                                    
                                                    input_shape=(n_steps_gru,n_features)))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense(1))
    
    #MLP
    if option == 4:
        
        X_train_final = X_train
        y_train_final = y_train

        n_features = X_train.shape[1]

        model = Sequential()
        
        model.add(Dense({{choice([37,16,96,22,95,35,21,59,12,88,86,30])}}, activation={{choice(['relu',
                                                            'tanh',
                                                            'sigmoid'])}}, 
                                                    
                                                    input_shape=(n_features,)))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense(1))
    
    #optimizer = SGD(learning_rate={{lognormal(0,0.003)}})
    model.compile(loss='mse', metrics=['mae'],optimizer='sgd')
    
    result = model.fit(X_train_final, y_train_final,
                       batch_size={{choice([1,2,4])}},
                       epochs=125,
                       verbose=0,
                       validation_split=0.1) #fit
    
    validation_mae = np.amin(result.history['val_mae'])
    print('Best validation mae of epoch:', validation_mae)
    
    return {'loss': validation_mae, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    
    best_run, best_model = optim.minimize(model=create_model, 
                                            data=data, 
                                            algo=tpe.suggest, 
                                            max_evals=2, 
                                            trials=Trials(),verbose=False)
    
    print(best_run)
    print(best_model)
    
    #Exportar

    model_json = best_model.to_json()
    
    with open('model2.json', 'w')as json_file:
        json_file.write(model_json)
    
    with open('results2.json','w') as results_file:
        results_file.write(str(best_run))
    