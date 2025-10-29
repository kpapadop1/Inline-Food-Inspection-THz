# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script containes the functions to build RNN models of a specific 
  architecture.
  
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script containes the functions to build RNN models of a specific 
  architecture.


SEE ALSO
  -
  
FILE
  .../build_rnn_model.py

ASSOCIATED FILES
  -

AUTHOR(S)
  K. Papadopoulos

DATE
  2022-October-01

LAST MODIFIED
  -

V1.0 / Copyright 2022 - Konstantinos Papadopoulos
-------------------------------------------------------------------------------
Notes
------

Todo:
------
- Make figure full-scale
  
-------------------------------------------------------------------------------
"""
#==============================================================================
#%% DEPENDENCIES
#==============================================================================
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, LSTM, GRU
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.optimizers import SGD, Adam
                                     
#==============================================================================
#%% FUNCTION DEFINITIONS
#==============================================================================

def build_rnn_model(type_rnn, sz_image, n_class, optimizer, learning_rate):
    
    # Define optimizer
    if optimizer == 'adam':
      optim = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
      optim = SGD(learning_rate=learning_rate)

    # Define RNN model of specific type using Keras' sequential definition
    model = Sequential()
    model.add(Input(shape=(sz_image, sz_image))) # seq_length, input_size
    if type_rnn == 0:
        
        print("Setting up simple RNN model...")
        model.add(SimpleRNN(units=128,
                            return_sequences=False,             # True (N_CLASS, 28, 128)
                            activation='tanh'))                 # ReLU doesnt lead to good results
        model.add(SimpleRNN(units=128,
                            return_sequences=False,             # True (N_CLASS, 28, 128)
                            activation='tanh'))                 # ReLU doesnt lead to good results
        
    elif type_rnn == 1:
        
        print("Setting up LSTM model...")
        model.add(LSTM(units=128,
                       return_sequences=False,                  # N_CLASS, N_UNITS
                       activation='tanh'))                      # ReLU doesnt lead to good results
        
    elif type_rnn == 2:
        
        print("Setting up GRU model...")
        model.add(GRU(units=128,
                      return_sequences=False,                   # N_CLASS, N_UNITS
                      activation='tanh')) 
        
    elif type_rnn == 3:
        
        print("Setting up bidirectional LSTM model...")
        model.add(Bidirectional(LSTM(units=128,                 # units=256
                                     return_sequences=False,    # N_CLASS, N_UNITS
                                     activation='tanh')))       # ReLU doesnt lead to good results
        
    model.add(Dense(units=n_class,                              # N_CLASS
                    activation='softmax'))
    
    # Compile model
    model.compile(optimizer=optim, 
                  loss=tf.losses.CategoricalCrossentropy(from_logits=False), 
                  metrics=['accuracy'])
    
    return model