# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script containes the functions to build and train a GAN model of a specific 
  architecture.
  
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script containes the functions to build and train a GAN model of a specific 
  architecture.


SEE ALSO
  -
  
FILE
  .../build_cgan_model.py

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
from keras.layers import (Input, Dense, Flatten, Conv2D, Reshape, Conv2DTranspose,
                          Concatenate, Embedding, LeakyReLU, BatchNormalization, 
                          Dropout)
from keras import backend as K

#==============================================================================
#%% FUNCTION DEFINITIONS
#==============================================================================

# Define function for building conditional gan model
def build_cgan_model(sz_image, learning_rate, latent_dim, n_classes):

  # Define function for building generator
  def build_generator(latent_dim, n_classes):

    # Define separate input path of generator for class label.
    label = Input(shape=(n_classes,),
                  name='Input-label')
    x_label = Embedding(input_dim=n_classes, 
                        output_dim=latent_dim,
                        name='Embedding-label')(label)
    x_label = Dense(units=7 * 7 * latent_dim,
                    name='Dense-label')(x_label)
    x_label = Reshape(target_shape=(7, 7, latent_dim),
                      name='Reshape')(x_label)
    
    # Define separate input path of generator for noise vector.
    noise = Input(shape=(latent_dim,),
            	    name='Input-noise')
    x_noise = Dense(units=128 * 7 * 7,
                    name='Dense-noise')(noise)
    x_noise = LeakyReLU(alpha=0.2)(x_noise)
    x_noise = Reshape((7, 7, 128))(x_noise)

    # Merge generated image and label
    x = Concatenate()([x_label, x_noise])

    # Add layers to reverse convolution in order to get original image dimensions 
    # and normalize in between
    x = Conv2DTranspose(filters=128, 
                        kernel_size=(4,4), 
                        strides=(2,2), 
                        padding='same', 
                        activation=LeakyReLU(alpha=0.2),
                        name='Conv2DTransp-1')(x) # Input image becomes a 14 x 14 image
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=128, 
                        kernel_size=(4,4), 
                        strides=(2,2), 
                        padding='same', 
                        activation=LeakyReLU(alpha=0.2),
                        name='Conv2DTransp-2')(x) # Input image becomes a 28 x 28 image
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=128, 
                        kernel_size=(4,4), 
                        strides=(2,2), 
                        padding='same', 
                        activation=LeakyReLU(alpha=0.2),
                        name='Conv2DTransp-3')(x) # Input image becomes a 56 x 56 image
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=128, 
                        kernel_size=(4,4), 
                        strides=(2,2), 
                        padding='same', 
                        activation=LeakyReLU(alpha=0.2),
                        name='Conv2DTransp-4')(x) # Input image becomes a 112 x 112 image
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=128, 
                        kernel_size=(4,4), 
                        strides=(2,2), 
                        padding='same', 
                        activation=LeakyReLU(alpha=0.2),
                        name='Conv2DTransp-5')(x) # Input image becomes a 224 x 224 image
    x = BatchNormalization()(x)
    output_img = Conv2D(filters=sz_image[-1], 
                        kernel_size=(7,7), 
                        activation='tanh', 
                        padding='same',
                        name='Conv2D')(x)

    # Set up generator
    generator = Model(inputs=[noise, label], 
                      outputs=output_img,
                      name='Generator')
    generator.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                      metrics=['accuracy'])
    
    return generator

  # Define function for building discriminator 
  def build_discriminator(input_shape, n_classes):
    
    # Define separate input path of discriminator for label
    # label = Input(shape=(1,),
    #               name='Input-label')
    label = Input(shape=(n_classes,),
                  name='Input-label')
    x_label = Embedding(input_dim=n_classes, 
                        output_dim=latent_dim,
                        name='Embedding-label')(label)
    # x_label = Dense(units=input_shape[0] * input_shape[1] * 1, # Attention: The third factor corresponds to the size of the label, which is 1.
    #                 name='Dense-label')(x_label)
    x_label = Dense(units=input_shape[0] * input_shape[1] * 1, # Attention: The third factor corresponds to the size of the label, which is 6.
                    name='Dense-label')(x_label)
    # x_label = Reshape(target_shape=(input_shape[0], input_shape[1], 1), # Attention: The last element corresponds to the size of the label, which is 1.
    #                   name='Reshape-label')(x_label)
    x_label = Reshape(target_shape=(input_shape[0], input_shape[1], n_classes), # Attention: The last element corresponds to the size of the label, which is 1.
                      name='Reshape-label')(x_label)
    
    # Define separate input path of discriminator for image.
    input_img = Input(shape=input_shape,
                    name='Input-image')

    # Merge image and label
    x = Concatenate()([input_img, x_label])

    # Add layers for convolution and a trailing dense layer in order to classify wether the 
    # image corresponds to the designated class or not (binary classification) while using
    # Dropout-layers in between
    x = Conv2D(filters=128, 
               kernel_size=(3, 3), 
               strides=(2, 2), 
               padding='same', 
               activation=LeakyReLU(alpha=0.2),
               name='Conv2D-1')(x)
    x = Dropout(rate=0.4,
                name='Dropout-1')(x)
    x = Conv2D(filters=128, 
               kernel_size=(3, 3), 
               strides=(2, 2), 
               padding='same', 
               activation=LeakyReLU(alpha=0.2),
               name='Conv2D-2')(x)
    x = Dropout(rate=0.4,
                name='Dropout-2')(x)
    x = Conv2D(filters=128, 
               kernel_size=(3, 3), 
               strides=(2, 2), 
               padding='same', 
               activation=LeakyReLU(alpha=0.2),
               name='Conv2D-3',)(x)
    x = Dropout(rate=0.4,
                name='Dropout-3')(x)
    x = Conv2D(filters=128, 
               kernel_size=(3, 3), 
               strides=(2, 2), 
               padding='same', 
               activation=LeakyReLU(alpha=0.2),
               name='Conv2D-4')(x)
    x = Dropout(rate=0.4,
                name='Dropout-4')(x)
    x = Conv2D(filters=128, 
               kernel_size=(3, 3), 
               strides=(2, 2), 
               padding='same', 
               activation=LeakyReLU(alpha=0.2),
               name='Conv2D-5')(x)
    x = Dropout(rate=0.4,
                name='Dropout-5')(x)
    x = Flatten(name='Flatten')(x)
    # out_label = Dense(units=1, 
    #               activation='sigmoid',
    #               name='Dense')(x)
    out_label = Dense(units=n_classes, 
                  activation='sigmoid',
                  name='Dense')(x)
    
    # Set up discriminator
    discriminator = Model(inputs=[input_img, label], 
                          outputs=out_label,
                          name='Discriminator')
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                          metrics=['accuracy'])

    return discriminator

  # Define function for building conditional GAN
  def build_cgan(generator, discriminator):

      # Turn discriminator to be non-trainable
      discriminator.trainable = False

      # Get inputs (noise and label) and outputs (image) from generator
      gen_noise, gen_label = generator.input
      gen_output = generator.output

      # Forward output of generator (image) and label input from generator to discriminator
      gan_output = discriminator([gen_output, gen_label])

      # define the conditional GAN model. 
      cgan = Model(inputs=[gen_noise, gen_label], 
                   outputs=gan_output,
                   name='cgan')
      cgan.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                   metrics=['accuracy'])
      return cgan

  # Set up generator, discriminator and conditional GAN
  generator = build_generator(latent_dim, n_classes)
  discriminator = build_discriminator(sz_image, n_classes)
  cgan = build_cgan(generator, discriminator)

  return generator, discriminator, cgan


# Function to get dataset samples
def get_dataset_samples(dataset, n_samples, n_classes):

  # Randomly select samples of number 'n_samples' out of passed dataset
  images, labels = dataset
  ix = np.random.randint(0, images.shape[0], n_samples)
  X, labels = images[ix], labels[ix]

  # Define a label indicating that this (randomly selected) subset is real.
  #y = np.ones((n_samples, 1))
  y = np.ones((n_samples, n_classes))

  return [X, labels], y

# Function to generate noise
def generate_noise(noise_size, n_samples, n_classes):
  
  # Generate random noise of normal distribution of size 'noise_size * n_samples' and reshape it
  x_input = np.random.randn(noise_size * n_samples)
  z_input = x_input.reshape(n_samples, noise_size)

  # Generate random labels
  labels = np.random.randint(low=0, high=n_classes, size=n_samples)

  return [z_input, labels]


# Function to generate fake samples
def generate_fake_samples(generator, latent_dim, n_samples, n_classes): 
  
  # Get noise and call function to generate images
  z_input, labels_input = generate_noise(latent_dim, n_samples, n_classes)   # TODO: labels_input must be of size N_BATCH/2, NUM_CLASSES
  images = generator.predict([z_input, labels_input])

  # Define a label indicating that this subset is fake.
  #y = np.zeros((n_samples, 1))
  y = np.zeros((n_samples, n_classes))

  return [images, labels_input], y


# Define function for training conditional gan model
#def train_cgan_model(sz_image, learning_rate, latent_dim, n_classes):
def train_cgan_model(generator, discriminator, cgan, dataset, n_classes, noise_size, n_epochs, n_batch):
  
  # Determine number of steps
  steps = int(dataset[0].shape[0] / n_batch)
  half_batch = int(n_batch / 2)

  # Enumerate epochs and over batches
  for e in range(n_epochs):
    for s in range(steps):

      # Discriminator training
      #-----------------------

      # Apply training to the discriminator using real dataset samples being randomly selected and update discriminator model weights
      [X_real, labels_real], y_real = get_dataset_samples(dataset, half_batch, n_classes)
      d_loss1, _ = discriminator.train_on_batch([X_real, labels_real], y_real)

      # Apply training to the discriminator using fake dataset samples being randomly selected and update discriminator model weights
      [X_fake, labels], y_fake = generate_fake_samples(generator, noise_size, half_batch, n_classes)
      d_loss2, _ = discriminator.train_on_batch([X_fake, labels], y_fake)

      # Generator training
      #-----------------------

      # Apply training to the generator after preparing points in latent space as input for the generator
      # and create inverted labels for the fake samples
      [z_input, labels_input] = generate_noise(noise_size, n_batch, n_classes)
      #y_gan = np.ones((n_batch, 1))
      y_gan = np.ones((n_batch, n_classes))

      # Apply training to the generator using discriminators' error (discriminator is not updated during this step)
      g_loss = cgan.train_on_batch([z_input, labels_input], y_gan)

      # Print losses while training this batch
      print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (e+1, s+1, steps, d_loss1, d_loss2, g_loss))

  # Save the generator model
  generator.save('cgan_generator.h5')

  return discriminator, cgan