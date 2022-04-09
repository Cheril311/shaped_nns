from keras.models import Sequential
from keras.regularizers import l1_l2
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Dense, Input, Dropout, Add, Activation, Embedding, LSTM, Bidirectional
import seaborn as sns
import tensorflow as tf
import pandas as pd

    

def shaped_bilstm(X,Y, para):
    

    model = Sequential()
    model.add(Bidirectional(LSTM(para['neuron_count'][0],return_sequences=True), input_shape=(para['dims'])))
    model.add(Dropout(para['dropout']))

    for i in range(para['layers'] - 2):

        model.add(Bidirectional(LSTM(para['neuron_count'][i+1],return_sequences=True)))
        model.add(Dropout(para['dropout']))

        model.add(Bidirectional(LSTM(para['neuron_count'][para['layers']])))
        model.add(Dropout(para['dropout']))

        model.add(Dense(para['neuron_last'], 
                activation=para['activation_out'],
                kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])))
    model.compile(loss=para['loss'],
              optimizer=para['optimizer'],
              metrics=para['metrics'])

    out = model.fit(X,Y, epochs=para['epoch'],steps_per_epoch = para['steps'],batch_size=para['batch_size'])
    return model, out

def bilstm(gen, para):
    

    model = Sequential()
    model.add(Bidirectional(LSTM(para['neuron_list'][0],return_sequences=True), input_shape=(para['dims'])))
    model.add(Dropout(para['dropout']))
    
    model.add(Bidirectional(LSTM(para['neuron_list'][1],return_sequences=True)))
    model.add(Dropout(para['dropout']))

    for i in range(2,para['layers'] - 2):

        model.add(Bidirectional(LSTM(para['neuron_list'][i+1])))
        model.add(Dropout(para['dropout']))


        model.add(Dense(para['neuron_last'], 
                activation=para['activation_out'],
                kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])))
    model.compile(loss=para['loss'],
              optimizer=para['optimizer'],
              metrics=para['metrics'])

    out = model.fit(gen, epochs=para['epoch'],steps_per_epoch=para['steps'])
    return model, out

def skip_bilstm(X,Y,para):
    input_shape = para['dims']
    input = Input(shape=input_shape)
    a = (Bidirectional(LSTM(32, return_sequences=True)))(input)

    x = (Bidirectional(LSTM(64, return_sequences=True)))(a) # main1 
    x = Dropout(para['dropout'])(x)
    a = (Bidirectional(LSTM(64, return_sequences=True)))(a) # skip1

    x = (Bidirectional(LSTM(64, return_sequences=True)))(x) # main1
    x = (Bidirectional(LSTM(64, return_sequences=True)))(x) # main1

    b = Add()([x,a]) # main1 + skip1

    x = (Bidirectional(LSTM(128, return_sequences=True)))(b) # main2
    x = Dropout(para['dropout'])(x)
    b = (Bidirectional(LSTM(128, return_sequences=True)))(b) # skip2

    x = (Bidirectional(LSTM(128, return_sequences=True)))(x) # main2
    x = Dropout(para['dropout'])(x)
    x = (Bidirectional(LSTM(128, return_sequences=True)))(x) # main2
    x = Dropout(para['dropout'])(x)

    c = Add()([b,x]) # main2 + skip2

    x = (Bidirectional(LSTM(256, return_sequences=False)))(c)
    x = Dropout(para['dropout'])(x)

    x = Dense(512, activation=para['activation'])(x)
    x = Dropout(para['dropout'])(x)
    x = Dense(128, activation=para['activation'])(x)
    x = Dropout(para['dropout'])(x)

    x = Dense(1, activation=para['activation_out'])(x)
    model = tf.keras.Model(input, x)
    
    model.compile(loss=para['loss'],
              optimizer=para['optimizer'],
              metrics=para['metrics'])

    out = model.fit(X,Y, epochs=para['epoch'],steps_per_epoch = para['steps'],batch_size=para['batch_size'])
    return model, out

