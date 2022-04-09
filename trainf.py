import pandas as pd
from T6.shape import shapes
from T6.models import mlp
from T6.models import regression
from T6.models import skipconn
from T6.models import skipshape
from T6.models import bi_lstm
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def trainer(X, Y, para):
    
    if para['test_size']:
        X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=para['test_size'],random_state = 42)
    try:
        dims = X.shape[1]
    except IndexError:
        dims = X_num

    para['dims'] = dims
    
    input_shape = (X.shape[1],)
    para['input_shape'] = input_shape
    
    if para['model'] is 'skip':
        model, history = skipconn(X_train, Y_train, para)
    if para['model'] is 'bilstm':
        model, history = bi_lstm(X_train, Y_train, para)
    
    if para['layers']:

        if para['layers'] == 1:
            para['shape'] = 'funnel'

        if para['neuron_max'] == 'auto' and dims >= 4:
            para['neuron_max'] = int(dims + (dims * 0.2))

        elif para['neuron_max'] == 'auto':
            para['neuron_max'] = 4

        para['neuron_count'] = shapes(para)

        if para['model'] is 'mlp':
            model, history = mlp(X, Y, para)
        
        if para['model'] is 'skip_shape':
            model, history = skipshape(X, Y, para)
        
        if para['model'] is 'regression':
            model, history = regression(X_train, Y_train, para)
        
    train_scores = model.evaluate(X_train, Y_train)
    test_scores=model.evaluate(X_test, Y_test)
    
    if Y_train.nunique()>=len(Y_train)//2:
        scores={'train_score':train_scores[0]**(1/2),'test_score':test_scores[0]**(1/2)}
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)
    else:
        scores={'train_score':train_scores[1],'test_score':test_scores[1]}
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)
        train_prediction = train_prediction.argmax(axis=-1)
        test_prediction = test_prediction.argmax(axis=-1)
    
    predictions={'train_predictions':train_prediction,'test_predictions':test_prediction}
    
    return scores,predictions,Y_train,Y_test
