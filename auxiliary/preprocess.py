# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script containes the preprocessing functions for the classification 
  task.
  
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script containes the preprocessing functions for the classification 
  task.


SEE ALSO
  -
  
FILE
  .../preprocess.py

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
import os
import numpy as np
import scipy as sp
import imghdr
import cv2
import tensorflow as tf
import keras.backend as K
from tensorflow.python.keras.backend import get_session
#from tools import plot_loop_state
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import exposure

#==============================================================================
#%% FUNCTION DEFINITIONS
#==============================================================================

def get_flops():
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=get_session().graph,
                                          run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops



# def get_flops_2(model):
#     if isinstance(model,(keras.functional.  keras.engine.functional.Functional, keras.engine.training.Model)):
#         run_meta=tf.compat.v1.RunMetadata()
#         opts=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#         from tensorflow.python.framework.convert_to_constants import (convert_variables_to_constants_v2_as_graph)
#         inputs=[tf.TensorSpec([1]+inp.shape[1:],inp.dtype) for inp in model.inputs]
#         real_model=tf.function(model).get_concrete_function(inputs)
#         frozen_func,_=convert_variables_to_constants_v2_as_graph(real_model)
#         flops=tf.compat.v1.profiler.profile(graph=frozen_func.graph,run_meta=run_meta,cmd="scope",options=opts)
#         return flops.total_float_ops



def convert_batchdataset_to_numpy(data):
    '''Convert batch dataset to numpy'''
    for x, y in data:
        x_data = x.numpy()
        y_data = y.numpy()
    return x_data, y_data

def set_denomination_computation(ML_METHOD, METHOD_NAME, MODEL_NAMES, TYPE_CNN, TYPE_RNN, TYPE_AE):
  '''Set denomination of current computation based on the main parameters'''
  MODEL_NAME = []
  if ML_METHOD == 0:
    MODEL_NAME = MODEL_NAMES[TYPE_CNN]
  elif ML_METHOD == 1:
    MODEL_NAME = MODEL_NAMES[TYPE_RNN]
  elif ML_METHOD == 2:
    MODEL_NAME = MODEL_NAMES[TYPE_AE]
  elif ML_METHOD == 3:
    MODEL_NAME = ''
  
  name_computation = "Class./" + METHOD_NAME + '/' + MODEL_NAME

    # if DL_METHOD == 0:
    #     name_computation += "CNN / "
    #     if TYPE_CNN == 0:
    #         name_computation += "Individual"
    #     elif TYPE_CNN == 1:
    #         name_computation += "LeNet-5"
    #     elif TYPE_CNN == 2:
    #         name_computation += "AlexNet"
    #     elif TYPE_CNN == 3:
    #         name_computation += "ZFNet"
    #     elif TYPE_CNN == 4:
    #         name_computation += "GoogLeNet"
    #     elif TYPE_CNN == 5:
    #         name_computation += "VGG-16"
    #     elif TYPE_CNN == 6:
    #         name_computation += "VGG-19"
    #     elif TYPE_CNN == 7:
    #          name_computation += "ResNet"
    # elif DL_METHOD == 1:
    #     name_computation += "RNN / "
    #     if TYPE_RNN == 0:
    #         name_computation += "Simple RNN"
    #     elif TYPE_RNN == 1:
    #         name_computation += "LSTM"
    #     elif TYPE_RNN == 2:
    #         name_computation += "GRU"
    #     elif TYPE_RNN == 3:
    #         name_computation += "Bi-LSTM"
    # elif DL_METHOD == 2:
    #     name_computation += "AE / "
    #     if TYPE_AE == 0:
    #         name_computation += "CAE"
    #     elif TYPE_AE == 0:
    #         name_computation += "CVAE"

  return name_computation

def check_images(data_dir):
    '''Check images by reading.'''
    
    # Define data directory
    image_exts = ['jpeg','jpg','bmp','png']
    
    # Print categories (names of contained folders)
    print("Categories:")
    for idx, image_class in enumerate(os.listdir(data_dir)):
        print(str(idx) + ": " + image_class)
    
    # Remove dodgy images by iterating through all and display message if any 
    # dodgy image was found
    n_images_all = 0
    class_name_list = os.listdir(data_dir)
    for image_class in class_name_list:
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            n_images_all+=1
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
    return n_images_all, class_name_list

def compression_svd(X, mode, percentage_keep, thd_coeff=1.0e-2):
    '''Function to compress image using SVD. The matrix is reconstructed based on the.
    INPUT:  X: [M x N] matrix (rows: Variables; columns: Observations)
            epislon: Whitening constant: prevents division by zero
    OUTPUT: ZCAMatrix: [M x M] matrix'''
    # Do for all images of set after changing the order of dimensions
    X_reordered = np.transpose(X, (0, 3, 1, 2))
    X_compressed = np.zeros_like(X_reordered)
    for i in range(len(X_reordered)):
      # Do for all channels
      for j in range(len(X_reordered[i])):

        U, s, Vh = sp.linalg.svd(X_reordered[i][j])
        sz = np.shape(X_reordered[i][j])
        sigma = np.zeros((sz[0], sz[1]))
        if mode == 'threshold_based':
          for k in range(min(sz[0], sz[1])):
              if s[k] > (np.max(s) - np.min(s))*thd_coeff:
                  sigma[k, k] = s[k]
              else:
                  break
        elif mode == 'percentage_based':
          for k in range(min(sz[0], sz[1])):
              if k <= int(min(sz[0], sz[1])*percentage_keep):
                  sigma[k, k] = s[k]
              else:
                  break
        X_comp = np.dot(U, np.dot(sigma, Vh))
        X_compressed[i][j] = X_comp
        # print("HarTimeRangeDopplerMaps() / Compression ratio: " + 
        #       str(np.count_nonzero(s)/np.count_nonzero(np.diagonal(sigma))))

      plot_loop_state(i,len(X_reordered),1/100)

    # Reorder to have again n_batch x n_rows x n_columns x 3
    X_compressed = np.transpose(X_compressed, (0, 2, 3, 1))
    return X_compressed

def normalization(X, coeff=255.0):
    '''Normalize data'''
    return X / coeff

def pca_zca_whitening(X, epsilon=1e-5, type='ZCA', rows_are_variables=True):
    '''Function to compute the PCA or ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix (rows: Variables; columns: Observations)
            epislon: Whitening constant: prevents division by zero
    OUTPUT: ZCAMatrix: [M x M] matrix'''
    # Do for all images of set after changing the order of dimensions
    X_reordered = np.transpose(X, (0, 3, 1, 2))
    X_whitened = np.zeros_like(X_reordered)
    for i in range(len(X_reordered)):
      # Do for all channels
      for j in range(len(X_reordered[i])):

        # Remove mean of each feature
        mean = np.mean(X_reordered[i][j], axis=0)
        X_centered = X_reordered[i][j] - mean

        # Covariance matrix [column-wise variables]: Sigma = (X - mu)' * (X - mu) / N. Here, its is assumed that rows correspond
        # to variables whereas columns correspond to observations ([M x N] matrix).
        sigma = np.cov(X_centered, rowvar=rows_are_variables) # [M x M]

        # Singular Value Decomposition. X = U * np.diag(S) * V
        # (U: [M x M] eigenvectors of sigma; S: [M x 1] eigenvalues of sigma; V: [M x M] transpose of U)
        U,S,V = np.linalg.svd(sigma)

        # Compute whitening matrix (M x M)
        if type == 'PCA':
          whitening_mat = np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)
        elif type == 'ZCA':
          # ZCA Whitening matrix: U * Lambda * U'
          whitening_mat = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))

        # Multiply with input matrix
        # y_whitened.append(whitening_mat @ y)
        X_whitened[i][j] = whitening_mat @ X_reordered[i][j]

      plot_loop_state(i,len(X_reordered),1/10)

    # Reorder to have again n_batch x n_rows x n_columns x 3
    X_whitened = np.transpose(X_whitened, (0, 2, 3, 1))
    return X_whitened

def eigenvalue_features(X, mode='min-avg-max', *args):
    '''Compute eigenvalues which serve as features for classical ML methods. The first 
      mode calculates the minimum, average and maximum eigenvalue while the second determines 
      the first n eigenvalues.'''
    # Preallocate
    if mode == 'min-avg-max':
      Y = np.zeros((len(X),3))
    elif mode == 'first_n':
      if len(args) == 1 and isinstance(args[0],int):
        Y = np.zeros((len(X),args[0]))

  # Cmpute eigenvalues for all images of set
    for i,x in enumerate(X):
      _,S,_ = np.linalg.svd(np.squeeze(x))
      if mode == 'min-avg-max':
        y = []
        Y[i,0] = np.min(S)
        Y[i,1] = np.mean(S)
        Y[i,2] = np.max(S)
      elif mode == 'first_n':
        Y[i,:] = S[0:args[0]]

      plot_loop_state(i,len(X), 1/10)

    return Y

def hog_features(X, sz_image):
  '''Compute history of oriented gradients (HOG) to be used as features.'''
  # Extract HOG features for training data
  hog_features = []
  for i,image in enumerate(X):
      # Calculate HOG features for each image
      fd, hog_image = hog(image.reshape((sz_image, sz_image)), orientations=9, pixels_per_cell=(56, 56), 
                          cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
      hog_features.append(fd)

      plot_loop_state(i,len(X), 1/10)

  hog_features = np.array(hog_features)

  return hog_features


def show_image_examples(data, class_name_list):
  '''Show some image examples.'''
  # Create iterator that returns numpy arrays in order to get only a batch 
  # instead of the whole data set for the plot operation.
  data_iterator = data.as_numpy_iterator()
  
  # Create consecutive batch out of data and save as tuple (batch[0].shape = (32,SZ_IMAGE,SZ_IMAGE,1) and batch[1].shape = (32,NUM_CLASSES))
  batch = data_iterator.next()
  
  # Create figure and plot consecutively some images
  fig, ax = plt.subplots(ncols=4, 
                        nrows=2, 
                        figsize=(20,20), 
                        gridspec_kw={'hspace':0.2, 'wspace':0.2})
  ax = ax.ravel()
  for idx, img in enumerate(batch[0][:8]):
      ax[idx].imshow(img.astype(int))
      ax[idx].title.set_text(class_name_list[np.argmax(batch[1][idx])])      