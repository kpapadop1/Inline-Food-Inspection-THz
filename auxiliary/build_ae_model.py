# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script containes the functions to build AE models of a specific 
  architecture.
  
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script containes the functions to build AE models of a specific 
  architecture.


SEE ALSO
  -
  
FILE
  .../build_ae_model.py

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
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import (Input, Dense, Flatten, Conv2D, 
                          MaxPooling2D, LeakyReLU, BatchNormalization, AveragePooling2D,
                          UpSampling2D, Reshape, Conv2DTranspose,
                          Lambda, Layer, Activation)
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
from keras import backend as K

#==============================================================================
#%% FUNCTION DEFINITIONS
#==============================================================================

def build_ae_model(type_ae, type_classifier, col_image, sz_image, optimizer, learning_rate, latent_dim, n_units_clf, n_classes):
    
  # Define color channel number
  if col_image == 'rgb':
    n_col = 3
  elif col_image == 'grayscale':
    n_col = 1

  # Define optimizer
  if optimizer == 'adam':
    optim = Adam(learning_rate=learning_rate)
  elif optimizer == 'sgd':
    optim = SGD(learning_rate=learning_rate)

  # Convolutional Autoencoder (CAE)
  if type_ae == 0:
      
    # Setup encoder layers
    input_img = Input(shape=(sz_image, sz_image, n_col))

    def encoder(input_img):
      x = Conv2D(filters=32, 
                  kernel_size=(3, 3),
                  padding='same',
                  name='Conv2D-1')(input_img)
      x = MaxPooling2D(pool_size=(2,2), 
                        padding='same',
                        name='MaxPooling2D-1')(x)
      x = LeakyReLU(alpha=0.2,
                    name='LeakyReLU-1')(x)
      x = BatchNormalization(axis=-1,
                              name='BatchNormalization-1')(x)
      x = Conv2D(filters=64, 
                  kernel_size=(3, 3),
                  padding='same',
                  name='Conv2D-2')(x)
      x = MaxPooling2D(pool_size=(2,2), 
                        padding='same',
                        name='MaxPooling2D-2')(x)
      x = LeakyReLU(alpha=0.2,
                    name='LeakyReLU-2')(x)
      x = BatchNormalization(axis=-1,
                              name='BatchNormalization-2')(x)
      x = Conv2D(filters=128, 
                  kernel_size=(3, 3),
                  padding='same',
                  name='Conv2D-3')(x)
      x = MaxPooling2D(pool_size=(2, 2), 
                        padding='same',
                        name='MaxPooling2D-3')(x)
      x = LeakyReLU(alpha=0.2,
                    name='LeakyReLU-3')(x)
      encodings = BatchNormalization(axis=-1,
                              name='BatchNormalization-3')(x)
      return encodings

    # Setup decoder layers
    def decoder(input_codings):
      x = Conv2DTranspose(filters=128, 
                  kernel_size=(3, 3),
                  padding='same',
                  name='Conv2DTransp-1')(input_codings)
      x = UpSampling2D(size=(2, 2),
                        name='UpSampling2D-1')(x)
      x = LeakyReLU(alpha=0.2,
                    name='LeakyReLU-4')(x)
      x = BatchNormalization(axis=-1,
                              name='BatchNormalization-4')(x)
      x = Conv2DTranspose(filters=64, 
                  kernel_size=(3, 3),
                  padding='same',
                  name='Conv2DTransp-2')(x)
      x = UpSampling2D(size=(2, 2),
                        name='UpSampling2D-2')(x)
      x = LeakyReLU(alpha=0.2,
                    name='LeakyReLU-5')(x)
      x = BatchNormalization(axis=-1,
                              name='BatchNormalization-5')(x)
      x = Conv2DTranspose(filters=32, 
                          kernel_size=(3, 3),
                          padding='same',
                          name='Conv2DTransp-3')(x)
      x = UpSampling2D(size=(2, 2),
                        name='UpSampling2D-3')(x)
      x = LeakyReLU(alpha=0.2,
                    name='LeakyReLU-6')(x)
      x = BatchNormalization(axis=-1,
                              name='BatchNormalization-6')(x)
      x = Conv2DTranspose(filters=3, 
                          kernel_size=(3, 3),
                          padding='same',
                          name='Conv2DTransp-4')(x)
      output_img = Activation('sigmoid',
                              name='Activation')(x)
      return output_img

    # Setup and compile autoencoder model
    cae = Model(inputs=input_img,
                outputs=decoder(encoder(input_img)),
                name='autoencoder')
    cae.compile(optimizer=optim, 
                loss=tf.keras.losses.mean_squared_error,
                metrics=tf.keras.metrics.mean_squared_error)
    
    # Classifier setup
    #-----------------

    # Define dense network
    def fc(enco):
      x = Flatten()(enco)
      x = Dense(units=n_units_clf, 
                activation='relu',
                name='Dense_classifier_1')(x)
      output = Dense(units=n_classes, 
                     activation='softmax',
                     name='Dense_classifier_2')(x)
      return output
      
    # Setup classifier model (compile at the caller, when the pretrained weights are gained)
    classifier = Model(inputs=input_img, 
                        outputs=fc(encoder(input_img)),
                        name='classifier')
  
    return cae, classifier

  # Convolutional Variational Autoencoder (CVAE)
  elif type_ae == 1:
      
    # Define sampling function for reparameterization
    def sampling(args):
          z_mean, z_log_var = args
          #  epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 
          #                                   K.int_shape(z_mean)[1])) # -> Substituted
          #  epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 
          #                                   LATENT_DIM))
          epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
          #epsilon = K.random_normal(shape=(K.shape(z_mean[0]), K.shape(z_mean[1])), mean=0.0, stddev=1.0)
          return z_mean + K.exp(0.5*z_log_var) * epsilon
      
    # Encoder setup
    #---------------

    # TODO:     
    # encoder_2 = MaxPooling2D((2,2),padding='same')(encoder_2)
    # encoder_2 = LeakyReLU(alpha=0.2)(encoder_2)
    # encoder_2 = BatchNormalization(axis=-1)(encoder_2)

    # Setup layers
    #  input_img = Input(shape=(SZ_IMAGE, SZ_IMAGE, 1), 
    #                    name='Input') # -> Substituted
    input_img = Input(shape=(sz_image, sz_image, n_col), 
                      name='Input')
    x = Conv2D(filters=32, 
              kernel_size=(3, 3),
              strides=(2, 2),
              padding='same',
              activation='relu',
              name='Conv2D-1')(input_img)
    x = Conv2D(filters=64, 
              kernel_size=(3, 3),
              strides=(2, 2),
              padding='same',
              activation='relu',
              name='Conv2D-2')(x)
    x = Conv2D(filters=128, 
              kernel_size=(3, 3),
              strides=(2, 2),
              padding='same',
              activation='relu',
              name='Conv2D-3')(x)
    shape_Conv2D_3 = K.int_shape(x)
    x = Flatten(name='Flatten')(x)
    shape_Flatten = K.int_shape(x)

    # Set up variation layer
    #x = Dense(32,activation='relu',name='Dense-1')(x) # -> Removed
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Create CVAE encoder
    encoder = Model(inputs=input_img, 
                    outputs=[z_mean, z_log_var, z], 
                    name='encoder')
    encoder.summary()
      
    # Decoder setup
    #---------------

    # TODO:
    # decoder_2 = UpSampling2D(size=(2,2))(decoder_2)
    # decoder_2 = LeakyReLU(alpha=0.2)(decoder_2)
    # decoder_2 = BatchNormalization(axis=-1)(decoder_2)

    # Set up layers
    latent_inputs = Input(shape=(latent_dim,), 
                          name='z_sampling')
    x = Dense(shape_Flatten[1],  #28 * 28 * 128
              activation='relu',
              name='Dense-2')(latent_inputs)
    x = Reshape((shape_Conv2D_3[1], shape_Conv2D_3[2], shape_Conv2D_3[3]),
                name='Reshape')(x)
    x = Conv2DTranspose(filters=128, 
                        kernel_size=(3, 3), 
                        strides=(2, 2), 
                        padding='same',
                        activation='relu',
                        name='Conv2DTransp-1')(x)
    x = Conv2DTranspose(filters=64, 
                        kernel_size=(3, 3), 
                        strides=(2, 2), 
                        padding='same',
                        activation='relu',
                        name='Conv2DTransp-2')(x)
    outputs = Conv2DTranspose(filters=3, 
                              kernel_size=(3, 3), 
                              strides=(2, 2), 
                              padding='same',
                              activation='sigmoid',
                              name='Conv2DTransp-3')(x)
    
    # Create CVAE decoder
    decoder = Model(inputs=latent_inputs, 
                    outputs=outputs, 
                    name='decoder')
    decoder.summary()

    # Apply decoder to latent variables
    z_decoded = decoder(z)
      
    # CVAE model setup
    #-----------------

    # # Set up the CVAE model
    # output_img = decoder(encoder(input_img)[2]) # Take z from [z_mean, z_log_var, z]
    # cvae = Model(inputs=input_img, 
    #              outputs=output_img, 
    #              name='cvae')
    # cvae.summary()
    
    # # Defining loss functions for CVAE model. The loss function contains a reconstruction term for the 
    # # autoencoder and Kullback-Leibler divergence as the regularization term for the encoder)
    # loss_reconstruction = tf.keras.losses.mean_squared_error(y_true=input_img, 
    #                                                          y_pred=output_img) * SZ_IMAGE * SZ_IMAGE * 3     # Check, if coefficients necessary!
    # loss_kullback_leibler = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # #loss_cvae = K.mean(loss_reconstruction + loss_kullback_leibler)
    # loss_cvae = loss_reconstruction + loss_kullback_leibler

    # # Compile the CVAE
    # cvae.add_loss(loss_cvae)
    # cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    #              metrics=[tf.keras.metrics.MeanSquaredError()])



    # Define custom loss
    # VAE is trained using two loss functions reconstruction loss and KL divergence
    # Let us add a class to define a custom layer with loss
    class CustomLayer(Layer):

        def vae_loss(self, x, z_decoded):
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)
            
            # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
            recon_loss = binary_crossentropy(x, z_decoded)
            
            # KL divergence
            kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(recon_loss + kl_loss)

        # Add custom loss to the class
        def call(self, inputs):
            x = inputs[0]
            z_decoded = inputs[1]
            loss = self.vae_loss(x, z_decoded)
            self.add_loss(loss, inputs=inputs)
            return x

    # Apply custom loss
    y = CustomLayer()([input_img, z_decoded])

    # Set up model and compile
    cvae = Model(input_img, y, name='cvae')
    cvae.compile(optimizer='adam', loss=None)
    cvae.summary()


    # # Set up the CVAE model using class definition derived from Keras' model type
    # class CVAE(Model):
    #     def __init__(self, encoder, decoder, **kwargs):
    #         super(CVAE, self).__init__(**kwargs)
    #         self.encoder = encoder
    #         self.decoder = decoder
    
    #     def train_step(self, data):
    #         if isinstance(data, tuple):
    #             data = data[0]

    #         with tf.GradientTape() as tape:
    #             z_mean, z_log_var, z = encoder(data)
    #             reconstruction = decoder(z)
    #             #reconstruction_loss = tf.reduce_mean(binary_crossentropy(data, reconstruction)) * 28 * 28
    #             reconstruction_loss = tf.keras.losses.mean_squared_error(y_true=input_img, 
    #                                                                      y_pred=output_img) * SZ_IMAGE * SZ_IMAGE * 3     # Check, if coefficients necessary!
    #             kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #             total_loss = reconstruction_loss + kl_loss
    #             grads = tape.gradient(total_loss, self.trainable_weights)
    #             self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    #         return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss,
    #         }

    # cvae = CVAE(encoder, decoder)
    # cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    #              metrics=[tf.keras.metrics.MeanSquaredError()])

    # Classifier setup
    #-----------------

    # Define the classifier
    output_class = Dense(units=n_classes, 
                         activation='softmax')(z_mean)
    classifier = Model(inputs=input_img, 
                        outputs=output_class,
                        name='classifier')
    classifier.compile(optimizer=optim,
                        metrics=['accuracy'],
                        loss='categorical_crossentropy')
    
    return cvae, classifier