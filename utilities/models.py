from keras.models import Sequential
from keras.regularizers import l1_l2
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Dense, Input, Dropout, Add, Activation, Embedding, LSTM, Bidirectional
import tensorflow as tf
import pandas as pd

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')

def mlp(X, Y, para):
    
    if Y.nunique() >= len(Y)//2:
        model = Sequential()
        model.add(Dense(para['neuron_count'][0],
                    input_dim=para['dims'],
                    activation=para['activation'],
                    kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])))
        model.add(Dropout(para['dropout']))

        for i in range(para['layers'] - 1):

            model.add(Dense(para['neuron_count'][i+1], 
                        activation=para['activation'],
                        kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])))
            model.add(Dropout(para['dropout']))

            model.add(Dense(para['neuron_last'], 
                    activation=para['activation_out'],
                    kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])))
        model.compile(loss=para['loss'],
                  optimizer=para['optimizer'],
                  metrics=para['metrics'])
    else:
        model = Sequential()
        model.add(Dense(para['neuron_count'][0],
                    input_dim=para['dims'],
                    activation=para['activation'],
                    kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])))
        model.add(Dropout(para['dropout']))

        for i in range(para['layers'] - 1):

            model.add(Dense(para['neuron_count'][i+1], 
                        activation=para['activation'],
                        kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])))
            model.add(Dropout(para['dropout']))

            model.add(Dense(para['neuron_last'], 
                    activation=para['activation_out'],
                    kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])))
        model.compile(loss=para['loss'],
                  optimizer=para['optimizer'],
                  metrics=para['metrics'])

    


    out = model.fit(X, Y, epochs=para['epoch'],batch_size=para['batch_size'])

    return model, out

def skipshape(X, Y, para):
    if Y.nunique() >= len(Y)//2:
        regul = regularizers.l2(para['l2'])
        inp = Input(para['input_shape'])
        x = inp
        for layer_size in para['neuron_count']:
            x = Dense(layer_size, activation=para['activation'], kernel_regularizer=regul)(x)
            if para['dropout']:
                x = Dropout(para['dropout'])(x)
        x = Dense(para['input_shape'][0])(x)
        x = Add()([x, inp])  # Skip connection
        x = Activation(activation=para['activation'])(x)
        out = Dense(1)(x)
        model = Model(inp, out)
        model.compile(optimizer=para['optimizer'], loss=para['loss'],metrics=['mse'])
    else:
        regul = regularizers.l2(para['l2'])
        inp = Input(para['input_shape'])
        x = inp
        for layer_size in para['neuron_count']:
            x = Dense(layer_size, activation=para['activation'], kernel_regularizer=regul)(x)
            if para['dropout']:
                x = Dropout(para['dropout'])(x)
        x = Dense(para['input_shape'][0])(x)
        x = Add()([x, inp])  # Skip connection
        x = Activation(activation=para['activation'])(x)
        out = Dense(para['neuron_last'])(x)
        model = Model(inp, out)
        model.compile(optimizer=para['optimizer'], loss=para['loss'],metrics=para['metrics'])
    out = model.fit(X, Y, epochs=para['epoch'],batch_size=para['batch_size'])
    
    return model,out

def skipconn(X,Y,para):
    if Y.nunique() >= len(Y)//2:
        regul = regularizers.l2(para['l2'])
        inp = Input(para['input_shape'])
        x = inp
        for layer_size in para['layer_sizes']:
            x = Dense(layer_size, activation=para['activation'], kernel_regularizer=regul)(x)
            if para['dropout']:
                x = Dropout(para['dropout'])(x)
        x = Dense(para['input_shape'][0])(x)
        x = Add()([x, inp])  # Skip connection
        x = Activation(activation=para['activation'])(x)
        out = Dense(1)(x)
        model = Model(inp, out)
        model.compile(optimizer=para['optimizer'], loss=para['loss'],metrics=para['metrics'])
    else:
        regul = regularizers.l2(para['l2'])
        inp = Input(para['input_shape'])
        x = inp
        for layer_size in para['layer_sizes']:
            x = Dense(layer_size, activation=para['activation'], kernel_regularizer=regul)(x)
            if para['dropout']:
                x = Dropout(para['dropout'])(x)
        x = Dense(para['input_shape'][0])(x)
        x = Add()([x, inp])  # Skip connection
        x = Activation(activation=para['activation'])(x)
        out = Dense(para['neuron_last'])(x)
        model = Model(inp, out)
        model.compile(optimizer=para['optimizer'], loss=para['loss'],metrics=para['metrics'])
    out = model.fit(X, Y, epochs=para['epoch'],batch_size=para['batch_size'])
    return model,out

    

def regression(X, Y,  para):

    
    model = Sequential()
    if Y.nunique() >= len(Y)//2:
            if para['reg_mode'] is 'linear':
                model.add(Dense(1, input_dim=X.shape[1]))
                model.compile(optimizer='rmsprop', metrics=['mse'], loss='mse')
    
            elif para['reg_mode'] is 'logistic':
                model.add(Dense(1, activation='sigmoid', input_dim=X.shape[1]))
                model.compile(optimizer='rmsprop', metrics=['mse'], loss='binary_crossentropy')
    
            elif para['reg_mode'] is 'regularized':
                kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])
                model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg, input_dim=X.shape[1]))
                model.compile(optimizer='rmsprop', metrics=['mse'], loss='binary_crossentropy')
    else:
            if para['reg_mode'] is 'linear':
                model.add(Dense(1, input_dim=X.shape[1]))
                model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='mse')
    
            elif para['reg_mode'] is 'logistic':
                model.add(Dense(1, activation='sigmoid', input_dim=X.shape[1]))
                model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='binary_crossentropy')
    
            elif para['reg_mode'] is 'regularized':
                kernel_regularizer=l1_l2(l1=para['l1'], l2=para['l2'])
                model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg, input_dim=X.shape[1]))
                model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='binary_crossentropy')

    out = model.fit(X, Y, epochs=para['epoch'])
    
    return model, out

def bi_lstm(X,Y,para):
    if para['prediction_len'] is 'auto':
        para['prediction_len'] = para['seq_len']

    dimensions = [1, para['seq_len'], para['dense_neurons'], 1]

    model = Sequential()
    model.add(Bidirectional(LSTM(dimensions[2],return_sequences=True),input_shape=([X.shape[1],1])))
    model.add(Dense(
        dimensions[3]))
    model.add(Dropout(para['dropout']))
    model.add(Activation(para['activation_out']))

    model.compile(loss=para['loss'], optimizer=para['optimizer'], metrics=para['metrics'])

    out = model.fit(X,
                        Y,
                        batch_size=para['batch_size'],epochs=para['epoch'],callbacks = [early_stopping])
    return model, out