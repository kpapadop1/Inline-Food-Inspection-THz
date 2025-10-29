# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script containes the functions to build CNN models of a specific 
  architecture.
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script containes the functions to build CNN models of a specific 
  architecture.


SEE ALSO
  -
  
FILE
  .../build_cnn_model.py

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
- Implement Hyperas (0.4.1): Hyperparameter tuning for neural networks
  
-------------------------------------------------------------------------------
"""
#==============================================================================
#%% DEPENDENCIES
#==============================================================================
import tensorflow as tf

from keras.models import (Model, Sequential)
from keras.layers import (Input, Dense, Dropout, Flatten, Conv2D, 
                          MaxPooling2D, AveragePooling2D, 
                          GlobalAveragePooling2D, ZeroPadding2D, Activation,
                          concatenate, BatchNormalization)
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras import backend as K

#==============================================================================
#%% FUNCTION DEFINITIONS
#==============================================================================

def build_cnn_model(type_cnn, activation_cnn_layer, activation_cnn_out,
                    sz_image, col_image, n_class, optimizer, learn_rate):
    
    # Define color channel number
    if col_image == 'rgb':
      n_col = 3
    elif col_image == 'grayscale':
      n_col = 1

    # Define optimizer
    if optimizer == 'adam':
      optim = Adam(learning_rate=learn_rate)
    elif optimizer == 'sgd':
      optim = SGD(learning_rate=learn_rate)

    # Define sequential CNN model of specific type
    model = Sequential()
    if type_cnn == 0:                               # Individual No.1
        
        model.add(Conv2D(filters=16, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer, 
                         input_shape=(sz_image, sz_image, n_col)))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=32, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=16, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(units=sz_image, 
                        activation=activation_cnn_layer))
        model.add(Dense(units=n_class, 
                        activation=activation_cnn_out))
        
        # Compile model
        model.compile(optimizer=optim, 
                      loss=CategoricalCrossentropy(), 
                      metrics=['accuracy'])

    elif type_cnn == 1: # Individual No. 2

        model.add(Conv2D(filters=16, # 8
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer,
                         kernel_regularizer=l2(1e-2),
                         bias_regularizer=l2(1e-4),
                         activity_regularizer=l2(1e-5),
                         kernel_initializer='he_normal',
                         input_shape=(sz_image, sz_image, n_col)))
        model.add(MaxPooling2D())
        model.add(Dropout(rate=0.4))
        model.add(Conv2D(filters=32, # 16
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer,
                         kernel_regularizer=l2(1e-2),
                         bias_regularizer=l2(1e-4),
                         activity_regularizer=l2(1e-5),
                         kernel_initializer='he_normal'))
        model.add(MaxPooling2D())
        model.add(Dropout(rate=0.4))
        model.add(Conv2D(filters=64, #32
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer,
                         kernel_regularizer=l2(0.01),
                         bias_regularizer=l2(1e-4),
                         activity_regularizer=l2(1e-5),
                         kernel_initializer='he_normal'))
        model.add(MaxPooling2D())
        model.add(Dropout(rate=0.4))
        model.add(Flatten())
        model.add(Dense(units=sz_image, 
                        activation=activation_cnn_layer))
        model.add(Dense(units=n_class, 
                        activation=activation_cnn_out)) 
        
        # Compile model
        model.compile(optimizer=optim, 
                      loss=CategoricalCrossentropy(), 
                      metrics=['accuracy'])

    elif type_cnn == 2: # Individual No. 3  

        model.add(Conv2D(filters=16, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer,
                         kernel_regularizer=l2(1e-2),
                         bias_regularizer=l2(1e-4),
                         activity_regularizer=l2(1e-5),
                         kernel_initializer='he_normal',
                         input_shape=(sz_image, sz_image, n_col)))
        model.add(MaxPooling2D())
        model.add(Dropout(rate=0.4))
        model.add(Conv2D(filters=32, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer,
                         kernel_regularizer=l2(1e-2),
                         bias_regularizer=l2(1e-4),
                         activity_regularizer=l2(1e-5),
                         kernel_initializer='he_normal'))
        model.add(MaxPooling2D())
        model.add(Dropout(rate=0.4))
        model.add(Conv2D(filters=64, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation=activation_cnn_layer,
                         kernel_regularizer=l2(0.01),
                         bias_regularizer=l2(1e-4),
                         activity_regularizer=l2(1e-5),
                         kernel_initializer='he_normal'))
        model.add(MaxPooling2D())
        model.add(Dropout(rate=0.4))
        model.add(Flatten())
        model.add(Dense(units=sz_image, 
                        activation=activation_cnn_layer))
        model.add(Dense(units=n_class, 
                        activation=activation_cnn_out)) 
        
        # Compile model
        model.compile(optimizer=optim, 
                      loss=CategoricalCrossentropy(), 
                      metrics=['accuracy'])

    elif type_cnn == 3:                               # LeNet-5 (1998)
    
        # Layer 1
        model.add(Conv2D(filters=6, 
                         kernel_size=(5,5), 
                         activation='tanh', 
                         input_shape=(sz_image, sz_image, n_col)))
        model.add(AveragePooling2D(pool_size=2, 
                                   strides=2, 
                                   padding='valid'))
        
        # Layer 2
        model.add(Conv2D(filters=16, 
                         kernel_size=(5,5), 
                         activation='tanh'))
        model.add(AveragePooling2D(pool_size=2, 
                                   strides=2, 
                                   padding='valid'))
        
        # Layer 3-4
        model.add(Flatten())
        model.add(Dense(units=120, 
                        activation='tanh'))
        model.add(Dense(units=84, 
                        activation='tanh'))
        model.add(Dense(units=n_class, 
                        activation='softmax'))
        
        # Compile model
        model.compile(optimizer=optim, 
                      loss=tf.losses.CategoricalCrossentropy(), 
                      metrics=['accuracy'])
        
    elif type_cnn == 4:                               # AlexNet (2012)
        
        # Layer 1
        model.add(Conv2D(filters=96, 
                         kernel_size=(11,11), 
                         strides=4,
                         padding='valid',
                         activation='relu', 
                         kernel_initializer= 'he_normal',
                         input_shape=(sz_image, sz_image, n_col)))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               padding='valid',
                               strides=2))
        
        # Layer 2
        model.add(Conv2D(filters=256, 
                         kernel_size=(5,5), 
                         strides=1,
                         padding='same',
                         activation='relu'),
                         kernel_initializer= 'he_normal')
        model.add(MaxPooling2D(pool_size=(3,3),
                               padding='valid',
                               strides=2))
        
        # Layers 3-5
        model.add(Conv2D(filters=384, 
                         kernel_size=(3,3), 
                         strides=1,
                         padding='same',
                         activation='relu'),
                         kernel_initializer= 'he_normal')
        model.add(Conv2D(filters=384, 
                         kernel_size=(3,3), 
                         strides=1,
                         padding='same',
                         activation='relu'),
                         kernel_initializer= 'he_normal')
        model.add(Conv2D(filters=384, 
                         kernel_size=(3,3), 
                         strides=1,
                         padding='same',
                         activation='relu'),
                         kernel_initializer= 'he_normal')
        model.add(MaxPooling2D(pool_size=(3,3),
                               padding='valid',
                               strides=(2,2)))
        
        # Layers 6-8
        model.add(Flatten())
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=4096, 
                        activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=4096, 
                        activation='relu'))
        model.add(Dense(units=1000, 
                        activation='relu'))
        model.add(Dense(units=n_class, 
                        activation='softmax'))
    
        # Compile model
        model.compile(optimizer=optim, 
                      loss=tf.losses.CategoricalCrossentropy(), 
                      metrics=['accuracy'])
        
    elif type_cnn == 5:                               # ZFNet (2013)
        
        # Layer 1
        model.add(Conv2D(filters=96, 
                         kernel_size=(7,7), 
                         strides=(2,2),
                         activation='relu',
                         input_shape=(sz_image, sz_image, n_col)))
        model.add(MaxPooling2D(pool_size=3, 
                               strides=2))
        
        # Layer 2
        model.add(Conv2D(filters=256, 
                         kernel_size=(5,5), 
                         strides=(2,2),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=3,
                               strides=2))
        
        # Layers 3-5
        model.add(Conv2D(filters=384, 
                         kernel_size=(3,3),
                         activation='relu'))
        model.add(Conv2D(filters=384, 
                         kernel_size=(3,3),
                         activation='relu'))
        model.add(Conv2D(filters=384, 
                         kernel_size=(3,3),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=3,
                               strides=2))
        
        model.add(Flatten())
        model.add(Dense(units=4096, 
                        activation='relu'))
        model.add(Dense(units=4096, 
                        activation='relu'))
        model.add(Dense(units=n_class, 
                        activation='softmax'))
        
        # Compile model
        model.compile(optimizer=optim, # tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        
    elif type_cnn == 6:                               # GoogLeNet (2014)
    
        # Place all layers until the first inception block
        input_layer = Input(shape = (sz_image, sz_image, n_col))
        X = Conv2D(filters=64, 
                   kernel_size=(7,7), 
                   strides=2, 
                   padding='valid', 
                   activation='relu')(input_layer)
        X = MaxPooling2D(pool_size=(3,3), 
                         strides=2)(X)
        X = Conv2D(filters=64, 
                   kernel_size=(1,1), 
                   strides=1, 
                   padding='same',
                   activation='relu')(X)
        X = Conv2D(filters=192, 
                   kernel_size=(3,3), 
                   padding='same',
                   activation='relu')(X)
        X = MaxPooling2D(pool_size=(3,3), 
                         strides=2)(X)

        # First inception block
        X = inception_block(X,
                            f1=64, 
                            f2_conv1=96, 
                            f2_conv3=128, 
                            f3_conv1=16, 
                            f3_conv5=32, 
                            f4=32)

        # 2nd inception block and max-pooling layer
        X = inception_block(X, 
                            f1=128, 
                            f2_conv1=128, 
                            f2_conv3=192, 
                            f3_conv1=32,
                            f3_conv5=96, 
                            f4=64)
        X = MaxPooling2D(pool_size=(3,3), 
                         strides=2)(X)

        # Third inception block
        X = inception_block(X,
                            f1=192, 
                            f2_conv1=96, 
                            f2_conv3=208, 
                            f3_conv1=16, 
                            f3_conv5=48, 
                            f4=64)

        # Extra network 1
        X1 = AveragePooling2D(pool_size=(5,5), 
                              strides=3)(X)
        X1 = Conv2D(filters=128, 
                    kernel_size=(1,1), 
                    padding='same', 
                    activation='relu')(X1)
        X1 = Flatten()(X1)
        X1 = Dense(units=1024, 
                   activation='relu')(X1)
        X1 = Dropout(0.7)(X1)
        X1 = Dense(units=5, 
                   activation='softmax')(X1)

  
        # Fourth inception block
        X = inception_block(X, 
                            f1=160, 
                            f2_conv1=112, 
                            f2_conv3=224, 
                            f3_conv1=24, 
                            f3_conv5=64, 
                            f4=64)

        # Fifth inception block
        X = inception_block(X, 
                            f1=128,
                            f2_conv1=128, 
                            f2_conv3=256, 
                            f3_conv1=24, 
                            f3_conv5=64,
                            f4=64)

        # Sixth Inception block
        X = inception_block(X,
                            f1=112,
                            f2_conv1=144, 
                            f2_conv3=288, 
                            f3_conv1=32, 
                            f3_conv5=64, 
                            f4=64)

        # Extra network 2
        X2 = AveragePooling2D(pool_size=(5,5), 
                              strides=3)(X)
        X2 = Conv2D(filters=128, 
                    kernel_size=(1,1), 
                    padding='same', 
                    activation='relu')(X2)
        X2 = Flatten()(X2)
        X2 = Dense(units=1024, 
                   activation='relu')(X2)
        X2 = Dropout(0.7)(X2)
        X2 = Dense(units=1000, 
                   activation='softmax')(X2)
  
        # Seventh inception block and max-pooling layer
        X = inception_block(X,
                            f1=256, 
                            f2_conv1=160, 
                            f2_conv3=320, 
                            f3_conv1=32, 
                            f3_conv5=128, 
                            f4=128)
        X = MaxPooling2D(pool_size=(3,3),
                         strides=2)(X)

        # Eight inception block
        X = inception_block(X, 
                            f1=256, 
                            f2_conv1=160, 
                            f2_conv3=320, 
                            f3_conv1=32, 
                            f3_conv5=128, 
                            f4=128)

        # Nineth inception block
        X = inception_block(X, 
                            f1=384, 
                            f2_conv1=192, 
                            f2_conv3=384, 
                            f3_conv1=48, 
                            f3_conv5=128, 
                            f4=128)

        # Add last layers (Global Average pooling layer, Dropout, output)
        X = GlobalAveragePooling2D(name='GAPL')(X)
        X = Dropout(0.4)(X)
        X = Dense(units=1000, 
                  activation='softmax')(X)
       
        # Set up model
        model = Model(input_layer, 
                      [X, X1, X2], 
                      name='GoogLeNet')
    
        # Compile model
        model.compile(optimizer=optim, 
                      loss=tf.losses.CategoricalCrossentropy(), 
                      metrics=['accuracy'])
    
    elif type_cnn == 7:                               # VGG-16 (2014)
    
        # Layers 1-2
        model.add(Conv2D(filters=64, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu', 
                         input_shape=(sz_image, sz_image, n_col)))
        model.add(Conv2D(filters=64, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 3-4
        model.add(Conv2D(filters=128, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=128, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 5-6
        model.add(Conv2D(filters=256, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=256, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 7-9
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 10-12
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 13-16
        model.add(Dense(units=25088, 
                        activation='relu'))
        model.add(Dense(units=4096, 
                        activation='relu'))
        model.add(Dense(units=4096, 
                        activation='relu'))
        model.add(Dense(units=n_class, 
                        activation='softmax'))
        
        # Compile model
        model.compile(optimizer=optim, 
                      loss=tf.losses.CategoricalCrossentropy(), 
                      metrics=['accuracy'])
        
    elif type_cnn == 8:                               # VGG-19 (2014)

        # Layers 1-2
        model.add(Conv2D(filters=64, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu', 
                         input_shape=(sz_image, sz_image, n_col)))
        model.add(Conv2D(filters=64, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 3-4
        model.add(Conv2D(filters=128, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=128, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 5-8
        model.add(Conv2D(filters=256, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=256, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=256, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=256, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 9-12
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 13-16
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(Conv2D(filters=512, 
                         kernel_size=(3,3), 
                         strides=1, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=2))
        
        # Layers 17-19
        model.add(Dense(units=4096, 
                        activation='relu'))
        model.add(Dense(units=4096, 
                        activation='relu'))
        model.add(Dense(units=n_class, 
                        activation='softmax'))

        # Compile model
        model.compile(optimizer=optim, 
                      loss=tf.losses.CategoricalCrossentropy(), 
                      metrics=['accuracy'])

    elif type_cnn == 9:                               # ResNet (2015)
        pass

    return model

# # Function name: inception_block
# # Purpose:       Build an inception block for the GoogLeNet-CNN
# #--------------------------------------------------------------
# def inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 

#   # First path
#   path_1 = Conv2D(filters=f1, 
#                   kernel_size=(1,1), 
#                   padding='same', activation='relu')(input_layer)

#   # Second path
#   path_2 = Conv2D(filters=f2_conv1, 
#                   kernel_size=(1,1), 
#                   padding='same', 
#                   activation='relu')(input_layer)
#   path_2 = Conv2D(filters=f2_conv3, 
#                   kernel_size=(3,3), 
#                   padding='same', 
#                   activation='relu')(path_2)

#   # Third path
#   path_3 = Conv2D(filters=f3_conv1, 
#                   kernel_size=(1,1),
#                   padding='same', 
#                   activation='relu')(input_layer)
#   path_3 = Conv2D(filters=f3_conv5, 
#                   kernel_size=(5,5), 
#                   padding='same', 
#                   activation='relu')(path_3)

#   # Fourth path
#   path_4 = MaxPooling2D((3,3),
#                         strides=(1,1),
#                         padding='same')(input_layer)
#   path_4 = Conv2D(filters=f4, 
#                   kernel_size=(1,1),
#                   padding='same',
#                   activation='relu')(path_4)

#   # Concatenate paths
#   output_layer = concatenate([path_1, path_2, path_3, path_4], axis = -1)

#   return output_layer

# # Class name: ResNet
# # Purpose:    Build the ResNet-CNN
# #--------------------------------------------------------------
# class ResNet:
#     @staticmethod
#     def residual_module(data, K, stride, chanDim, red=False,
#                         reg=0.0001, bnEps=2e-5, bnMom=0.9):
        
#         # The shortcut branch of the ResNet module should be
#         # initialize as the input (identity) data
#         shortcut = data
    
#        	# the first block of the ResNet module are the 1x1 CONVs
#         bn1 = BatchNormalization(axis=chanDim, 
#                                     epsilon=bnEps,
#                                     momentum=bnMom)(data)
#         act1 = Activation("relu")(bn1)
#         conv1 = Conv2D(int(K*0.25), 
#                        (1,1), 
#                        use_bias=False, 
#                        kernel_regularizer=l2(reg))(act1)
      
#        	# The second block of the ResNet module are the 3x3 CONVs
#         bn2 = BatchNormalization(axis=chanDim, 
#                                  epsilon=bnEps, 
#                                  momentum=bnMom)(conv1)
#         act2 = Activation("relu")(bn2)
#         conv2 = Conv2D(int(K*0.25), 
#                        (3,3), 
#                        strides=stride, 
#                        padding="same", 
#                        use_bias=False, 
#                        kernel_regularizer=l2(reg))(act2)
       
#        	# The third block of the ResNet module is another set of 1x1 CONVs
#         bn3 = BatchNormalization(axis=chanDim, 
#                                  epsilon=bnEps,
#                                  momentum=bnMom)(conv2)
#         act3 = Activation("relu")(bn3)
#         conv3 = Conv2D(K, 
#                        (1,1), 
#                        use_bias=False, 
#                        kernel_regularizer=l2(reg))(act3)
       
#        	# If we are to reduce the spatial size, apply a CONV layer to the 
#         # shortcut
#         if red:
#        	  shortcut = Conv2D(K,
#                             (1,1),
#                             strides=stride,
#                             use_bias=False, 
#        	   		              kernel_regularizer=l2(reg))(act1)
       
#        	# Add together the shortcut and the final CONV
#         x = add([conv3, shortcut])
       
#        	# Return the addition as the output of the ResNet module
#         return x
       
#     @staticmethod
#     def build(width, height, depth, classes, stages, filters, 
#               reg=0.0001, bnEps=2e-5, bnMom=0.9):

#        	# initialize the input shape to be "channels last" and the channels 
#         # dimension itself
#         inputShape = (height, width, depth)
#         chanDim = -1
       
#        	# If we are using "channels first", update the input shape
#        	# and channels dimension
#         if K.image_data_format() == "channels_first":
#        	  inputShape = (depth, height, width)
#        	  chanDim = 1
            
#         # Set the input and apply BN
#         inputs = Input(shape=inputShape)
#         x = BatchNormalization(axis=chanDim, 
#                                epsilon=bnEps,
#                                momentum=bnMom)(inputs)
  
#   		# Apply CONV => BN => ACT => POOL to reduce spatial size
#         x = Conv2D(filters[0],
#                    (5,5), 
#                    use_bias=False,
#                    padding="same",
#                    kernel_regularizer=l2(reg))(x)
#         x = BatchNormalization(axis=chanDim, 
#                                epsilon=bnEps,
#                                momentum=bnMom)(x)
#         x = Activation("relu")(x)
#         x = ZeroPadding2D((1,1))(x)
#         x = MaxPooling2D((3,3), strides=(2, 2))(x)
          
#       	# Loop over the number of stages
#         for i in range(0, len(stages)):
#             # Initialize the stride, then apply a residual module
#             # sed to reduce the spatial size of the input volume
#             stride = (1, 1) if i == 0 else (2, 2)
#             x = ResNet.residual_module(x, 
#                                        filters[i + 1], 
#                                        stride,
#                                        chanDim, 
#                                        red=True, 
#                                        bnEps=bnEps, 
#                                        bnMom=bnMom)
      
#         # Loop over the number of layers in the stage
#         for j in range(0, stages[i] - 1):
#             # Apply a ResNet module
#             x = ResNet.residual_module(x, 
#                                        filters[i + 1],
#                                        (1, 1), 
#                                        chanDim, 
#                                        bnEps=bnEps, 
#                                        bnMom=bnMom)
             
#        	# Apply BN => ACT => POOL
#         x = BatchNormalization(axis=chanDim, epsilon=bnEps,
#        			momentum=bnMom)(x)
#         x = Activation("relu")(x)
#         x = AveragePooling2D((8, 8))(x)
          
#        	# Softmax classifier
#         x = Flatten()(x)
#         x = Dense(classes, kernel_regularizer=l2(reg))(x)
#         x = Activation("softmax")(x)
      	
#        	# Create the model
#         model = Model(inputs, x, name="resnet")
      	
#        	# return the constructed network architecture
#         return model