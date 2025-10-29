# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script defines the global class for the Inline Food Inspection experiments called 'IFI'.
  
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script defines the global class for the Inline Food Inspection experiments called 'IFI'.


SEE ALSO
  -
  
FILE
  .../IFI.py

ASSOCIATED FILES
  -

AUTHOR(S)
  K. Papadopoulos
  J. Brüninghaus
  H. Al-Joumaa

DATE
  2024-December-19

LAST MODIFIED
  -

V1.0 / Copyright 2025 - Konstantinos Papadopoulos
-------------------------------------------------------------------------------
Notes
------

Todo:
------

-------------------------------------------------------------------------------
"""
# region dependencies

# Miscellaneous
import math
import asyncio
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import socket
import os
import sys
import subprocess
import json
import warnings
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# THz
from teraFAST import processor as tp
from teraFAST import worker as tw
import numpy as np
from PIL import Image, ImageEnhance
import sys, os
import pickle
from datetime import datetime
import keyboard
import matlab.engine
from auxiliary.THz import THz

# FMCW
from auxiliary.Fmcw import Fmcw

# CV/MV
from pypylon import pylon
import cv2
from auxiliary.MV import MV
import skimage as sk
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import adapted_rand_error
from skimage.metrics import normalized_mutual_information
from skimage.metrics import variation_of_information
from image_similarity_measures.evaluate import evaluation
from ssim import SSIM
import pywt

# ML/DL
import tensorflow as tf
from tensorflow import keras
#from keras import ops
from keras.layers import (Input, Conv2D, Flatten, Dense, 
                          Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, UpSampling2D, Concatenate, MultiHeadAttention, Add, Subtract, Multiply)
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import CategoricalCrossentropy
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc

# endregion


#==============================================================================
#%% CLASS DEFINITION
#==============================================================================

class IFI:
  """This class implements all classes and methods for the application of Inline Food Inspection techniques.
  
  The parameter object (dict) has the following structure:

  trigger_steps (list):
    Number of trigger steps (mode not implemented yet) 
  udp_ip (str): 
    Host IP
  udp_port (str):
    Port No. of UDP communication partner (trigger system)
  udp_max_n_byte (int):
    Number of maximum bytes of UDP message
  thz (dict):
    THz parameters with the following structure:
  
    n_images 8INT9:
      Number of images to acquire
    mode Iint):
      Mode: 0 -> without config; 1 -> Record and save to config file; 2 -> Load config file
    config_name (str): 
      Name of configuration file
    rate (int):
      Rate in lines/s
    save (bool):
      Save to image directly
    t_sleep (float):
      Sleep time between each image
    init:
      Initialization parameters with the following structure:
      
      LOAD_CONFIG (bool):
        Flag: load configuration
      SAVE_CONFIG (bool): 
        Flag: save configuration
      MODE_BACKGROUND (str):
        Mode for background handling ('RECORD'; 'SELECT'; 'SET'; see TeraFAST Python API Documentation)
      MODE_NORM (str):
        Mode for normalization handling ('RECORD'; 'SELECT'; 'SET'; see TeraFAST Python API Documentation)
      MODE_DIFFERENCE (bool): 
        Turn difference mode on or off (default: True)
      SET_SYNC_OUT (bool):
        Turn sync out on (default: True)
      SET_EXTERNAL_SYNC (bool):
        Turns the external synchronization mode on or off
      EXTERNAL_SYNC_EDGE (bool):
        Trigger type for external synchronization (True: Rising edge; False: falling edge, None: keep unchanged)
      SYNC_DIVIDER (int):
        Frequency divider for external synchronization (1...32768)
      SYNC_HOLDOFF (int):
        Holdoff parameter for external synchronization (0...15)
      LOAD_CONFIG_PATH' (str):
        Path for config to load, if selected
      SAVE_CONFIG_PATH (bool):
        Path for config to save, if selected
      FRAME_LENGTH (int):
        Number of lines returned per frame
      RATE (int):
        Acquisition rate in lines/second
      BACKGROUND_IDX (int):
        Index for backgrond selection out of a list
      BACKGROUND_DATA (numpy array):
        Prerecorded background data
      BACKGROUND_RECORD_COUNTS (int):
        Number of records for background measurement (default: 30)
      NORM_IDX (int)
        Index for backgrond selection out of a list
      NORM_DATA_MASK (tupel):
        Prerecorded normalization data and mask (latter used for excluding specific faulty pixels from normalization)
      THRESHOLD (float):
        Threshold value for the SNR to declare pixels unusable with recorded normalization (default: 10.0)
      ACCUMULATION_ON (bool):
        Turn accumulation on or off
      ACCUMULATION_LENGTH (int):
        Accumulation length (0...100)
      BLACK (float):
        Black value for brightness and contrast adjustment
      WHITE (float):
        White value for brightness and contrast adjustment
      GAMMA (float):
        Gamma value
      SMOOTHNESS (int):
        Smoothness parameter (0...100) for gaussian blurring or median smoothing
                             
  mv (dict): 
    MV parameters with the following structure:

    mode (str):
      Mode: foreground loop
    n_images (int):
      Number of images to acquire
    timeout_ms (int):
      Timeout in ms
    show_stats (bool):
      Show statistics flag
    idx_device (int): 
      Device index
    show_one_image (bool)
      Show one image flag
    idx_show_one_image (int):
      Index of single image to show

  Attributes:

  Methods:
    thz_measurement: 
      Asynchronous function that applies the methods of the THz system
    fmcw_measurement:
      Asynchronous function that applies the measurement operations of the FMCW system.
    mv_measurement:
      Asynchronous function that applies the measurement operations of the MV system (default: None).
    
    matlab_mmwstudio_startup:
      Apply startup of MATLAB and TI's mmWave Studio.
    configure_udp_socket: 
      This functions configures the UDP socket, which is required for the communication with the measurement trigger.
    listen_for_udp_trigger:
      This functions listens for UDP-based trigger and sets a flag, if a valuid message was received.

  """

  t_0 = None
  MEASUREMENT_TECHNOLOGIES = ['thz', 'thz_raw', 'mv']
  NAME_DATASET_FOLDERS = ['train', 'val_regular', 'val_anomalous']
  NAME_INPUT_OUTPUT_FOLDERS = ['input', 'output']

  def __init__(self):
    pass

  # region Asynchronous methods for measurements

  def thz_measurement(self, thz_obj, parms_thz=None):
    """Asynchronous function that applies the methods of the THz system (default: None).

    Args:
      thz_obj (THz object):
        Terahertz measurement instance
      parms_thz (dict): 
        Terahertz measurement parms

    Returns:
      img_thz_list (2D numpy array): 
        Terahertz image
      source (specific object): 
        Terasense source object (camera)

    Raises:
    """

    # Call THz measurement function
    while True:

      if (self.t_0 is not None) and \
      ((time.time() - self.t_0) >= parms_thz['trigger_time']):
        
        t_0 = time.time()

        if parms_thz['active']:
          # print('THz measurement starting....')
          # (img_thz, 
          #  data_thz, 
          #  source, 
          #  convert) = thz_obj.execute_measurement_simple(source, 
          #                                         convert,
          #                                         n_images=parms_thz['n_images'], 
          #                                         save=parms_thz['save'],
          #                                         t_sleep=parms_thz['t_sleep'])
          img_thz = thz_obj.execute_measurement_simple()
        else:
          print('THz measurement deactivated.')
          img_thz = 0

        print(f"THz duration: {time.time() - t_0} s.")
        break

    return img_thz #, data_thz, source, convert

  def fmcw_measurement(self, me, parms_fmcw=None):
    """Asynchronous function that applies the measurement operations of the FMCW system.

    Args:
      me (specific object): 
        MATLAB engine

    Returns:
      results (list):
        Processed results
      me (specific object): 
        MATLAB engine

    Raises:
    """

    # Call FMCW measurement function
    results = None
    while True:

      if (self.t_0 is not None) and \
        ((time.time() - self.t_0) >= parms_fmcw['trigger_time']):

        t_0 = time.time()
        
        if parms_fmcw['active']:
          print('FMCW measurement starting....')
          results = me.startmmwstudio()
        else:
          print('FMCW measurement deactivated.')
          results = None

        print(f"FMCW duration: {time.time() - t_0} s.")
        break

    return results, me
  
  def mv_measurement(self, mv_obj, camera, devices, parms_mv=None):
    """Asynchronous function that applies the measurement operations of the MV system (default: None).

    Args:
      mv_obj (MV object): 
        MV instance
      camera (specific object):
        Basler camera object
      devices (list):
        devices
      parms_mv (dict):
        Parameters of MV system

    Returns:
      img_mv (numpy array):
        Image acquired by Basler camera
      camera (specific object): 
        Basler camera object

    Raises:
    """

    # Call MV measurement function
    img_mv = None
    while True:

      if (self.t_0 is not None) and \
        ((time.time() - self.t_0) >= parms_mv['trigger_time']):

        t_0 = time.time()

        if parms_mv['active']:
          print('MV measurement starting....')
          (img_mv, 
          camera) = mv_obj.grab_software_triggered_camera_basler(parms_mv, camera, devices)
          camera = mv_obj.basler_camera_disconnect(camera)
        else:
          print('MV measurement deactivated.')
          img_mv = None

        print(f"MV duration: {time.time() - t_0} s.")
        break

    return img_mv, camera

  # endregion

  # region Analysis

  def analyze_image_similarity(self, folder_source=None, folder_results=None,name_file_reference=None, name_file_contaminated=None, mode=None, delta_list=None):
    '''This function computes similarity metrics for every subfolder based on two images whose names match a definition.
    
    Args:
    folder_source (str):
      Source folder containing dataset (default: None)
    folder_results (str):
      Results folder (default: None)
    name_file_reference (str):
        Name of the reference image (default: None)
    name_file_contaminated (str):
        Name of the reference image (default: None)
    mode (str):
      Mode: 'cwssim_fsim_chi2' -> Calculate CW-SSIM, FSIM, CHI2
            'ssim_mse_nmi_re_voi' -> Calculate SSIM, MSE, NMI, RE, VOI
            'experimental' -> Experimental similarity indices (default: None)
    delta_list (list of float):
      Values for binarization threshold (default: None)

    Returns:
      folder_results_date (str): 
        Relative path of results folder since all folder name are expanded by a date and time string.

    Raises:
      ValueError("Unknown mode")
    
    '''
    folder_results_date = None
    if mode == 'common_similarity_metrics':
      folder_results_date = self.calculate_common_similarity_metrics(folder_source, folder_results, delta_list)

    elif 'experimental':
      for dirpath, dirnames, filenames in os.walk(folder_source):
          if name_file_reference in filenames and name_file_contaminated in filenames:
              image_path_reference = os.path.join(dirpath, name_file_reference)
              image_path_contaminated = os.path.join(dirpath, name_file_contaminated)
              folder_results_date = self.calculate_similarity_index
              (image_path_reference, image_path_contaminated, mode)
    else:
      raise ValueError("Unknown mode")
    
    return folder_results_date

  def classify_detect(self, path_model=None, path_dataset=None):
    '''Apply analyzing operations such as classification and anomaly detection. 
    
    As input, datasets of three images are required (THz, FMCW, MV) along with their tag file (YOLO structure).

    Args:
      path_model (str):
        Path to saved model (if available) (default: None)
      path_dataset (str):
        Path to global dataset folder. This folder should contain three separate folders for the train data, regular validation and anomalous valdation data.

    Returns:

    Raises:
    '''

    # region preprocessing

    input_data = self.get_input_data(path_dataset)

    # TEMP: Peak into images
    #ifi_2.peek_results_pickled('measurements_ifi\\ifi_2025_04_25-09_52_15')
    ifi_2.peek_results_input(input_data, 'anomalous', idx=8)

    # Check HW in order to predict training duration
    print(f"Number of available GPUs: {len(tf.config.list_physical_devices('GPU'))}")

    # Set up consistent datasets
    dataset = self.load_dataset(path_dataset)
    x_train = dataset[0]
    y_train = dataset[1]
    x_val_regular = dataset[2]
    y_val_regular = dataset[3]
    x_val_anomalous = dataset[4]
    y_val_anomalous = dataset[5]

    # Adjust_image_quality
    image_list = self.adjust_image_quality(image_list, 
                                           coeff_brightness=1.0, coeff_contrast=1.0, 
                                           coeff_gamma=1.0)

    #TODO
    cae_input = []

    # endregion

    # region main operation

    # Setup model
    if path_model is None:
      cae_model = self.set_cae_model(cae_encoder_input=cae_input, 
                                    input_dim=(256, 256, 3), 
                                    z_dim=512, 
                                    type_loss='mse')
      cae_model.save("cae_model")
    else:
      cae_model = tf.keras.models.load_model("cae_model")
    
    # Determine early stopping conditions and fit model to data
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=13, restore_best_weights=True)
    history = cae_model.fit(x_train, 
                            y_train, 
                            batch_size=32,
                            epochs=40, 
                            validation_data=(x_val_regular, y_val_regular), shuffle=True)
    K.clear_session()

    # endregion

    # region postprocessing

    # Calculate threshold
    thd = self.calc_class_threshold(self, cae_model, 
                                    [x_val_regular, x_val_anomalous], 
                                    [y_val_regular, y_val_anomalous])

    # Predict
    # TODO

    # endregion

  def calc_class_threshold(self, model=None, x=None, y=None):
    '''Calculate threshold for anomaly detection based on lists of regular and anomalous datasets.
    
    Args:
      model (Model):
        Model for anomaly detection (default: None)
      x (List):
        Features of regular and anomalous datasets (in that order) as lists (default: None)
      y (List):
        Features of regular and anomalous datasets (in that order) as lists (default: None)

    Returns:
      thresh (float):
        Threshold for separation of regular and anomalous data

    Raises:
    '''
    
    # Determine losses for anomalous data and regular data
    losses = [h for h in range(2)]
    for h in range(len(losses)):

      for i in range(x[h].shape[0]):
        loss = model.evaluate(x[h][i][None, :, :, :], 
                              y[h][i][None, :, :, :], 
                              verbose=0)
        losses[h].append(loss)
        K.clear_session()

      # Determine threshold using Otsu's method
      losses_all = np.concatenate(losses)
      thresh = threshold_otsu(losses_all)

    return thresh

  # endregion

  # region image load

  def load_dataset(self, path_dataset):
    '''Load a specific dataset using its path.
    
    Args:
      path_dataset (str):
        Path to dataset folder.

    Returns:
      dataset (List of lists of 3D numpy arrays):
        Images
    
    Raises:
      e: Error ocurred during image loading.
    '''
    dataset = [[None for i in range(len(self.NAME_INPUT_OUTPUT_FOLDERS))] for j in range(len(self.NAME_DATASET_FOLDERS))]


    # Go through all three folders of the dataset and both subfolders and load images based on this scheme: (x_train, y_train, x_val_regular, y_val_regular, x_val_anomalous,  y_val_anomalous)
    for _, dirs, _ in os.walk(path_dataset):
      for index_dir, dirname in enumerate(dirs):
        if dirs in self.NAME_DATASET_FOLDERS:

          for _, dirs_1, _ in os.walk(os.path.join(path_dataset, dirs)):
            for index_dir_1, dirname in enumerate(dirs_1):
              if dirs_1 in self.NAME_INPUT_OUTPUT_FOLDERS:

                for filename in os.listdir(os.path.join(path_dataset, dirs, dirs_1)):
                  if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_path = os.path.join(path_dataset, filename)
                    try:
                      img_array = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
                      img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                      dataset[index_dir][index_dir_1].append(img_array.astype("float32") / 255.0)
                    except Exception as e:
                      print(f"Error ocurred while image loading {filename}.")

    return dataset

  def get_input_data(self, base_folder, subfolder_names=('normal','anomalous'), image_denom=['mv.png','thz_man.png'], target_size=(2500, 2500)):
    '''Reads images in their subfolders for a given folder and aggregates them while handling normal and anomalous cases separately.
    
    Args:
      base_folder(str): 
        Base folder path containing folders of individual measurements
      subfolder_names (tupel):
        Names of the two subfolder containing normal and anomalous data (in exactly that order)
      image_denom (list):
        Image denomination (default: ['mv.png','thz.png','thz_raw.png'])
      target_size (tupel):
        Target size after resizing (default: (2500, 2500))
      
    Returns:

    Raises:
    '''
    # Select device index:
    #   0: 'IWR6843_AOP_REV_G'
    #   1: 'IWR443_BOOST'
    #   2: 'MMWCAS_RF_EVM'
    device_index = 2
    # Prepare radar object for data processing
    with open('parms_fmcw.json', 'r') as file:
        parms_fmcw = json.load(file)

    combined_images = [[],[]]

    # Go through image main folders (normal, anomalous)
    for folder in os.listdir(base_folder):
      folder_path = os.path.join(base_folder, folder)
      
      # Process folder containing normal and anomalous images ...
      idx_type = None
      if os.path.isdir(folder_path) and folder=='normal':
        idx_type = 0
      elif os.path.isdir(folder_path) and folder=='anomalous':
        idx_type = 1

      # ... and their subfolders.
      if idx_type is not None:
        for subfolder_path in os.listdir(folder_path):
          images = []
          for idx in range(len(image_denom)):

            try:
              
              if idx == 0:
                file_path = os.path.join(folder_path, 
                        subfolder_path, 
                        image_denom[idx])
                with Image.open(file_path) as img:
                  images.append(cv2.resize(np.array(img), target_size))
              elif idx == 1:
                file_path = os.path.join(folder_path, 
                        subfolder_path, 
                        image_denom[idx])
                with Image.open(file_path) as img:
                  images.append(np.rot90(cv2.resize(np.array(img), target_size), k=1))
              elif idx == 2:

                # Load variables from the JSON file
                with open('parms_fmcw.json', 'r') as file:
                  parms_fmcw = json.load(file)

                fmcw_data_path = os.path.join(folder_path, subfolder_path) 

                fmcw_1 = Fmcw(parms_fmcw['parms_board'][device_index], 
                              parms_fmcw['parms_processing'], 
                              parms_fmcw['parms_chirp_profile'],
                              parms_fmcw['parms_target'],
                              parms_fmcw['parms_show'],
                              parms_fmcw['parms_sar'],
                              measurement_folder_path=fmcw_data_path,
                              parameter_file_path=fmcw_data_path + '\\Cascade_Capture.mmwave.json',
                              calibration_file_path='measurements_ifi\\calibration\\calibrateResults_high_20240829T112982_Calibration_1.mat')

                img = fmcw_1.sar_processing_chain(folder_path=fmcw_data_path,
                                                  json_file_path=fmcw_data_path + '\\Cascade_Capture.mmwave.json',
                                                  calibration_file_path='measurements_ifi\\calibration\\calibrateResults_high_20240829T112982_Calibration_1.mat')
                
                images.append(cv2.resize(np.array(img), target_size))

            except Exception as e:
              print(f"Error loading {file_path}: {e}")

          if images:
            combined_images[idx_type].append(images)

    return combined_images

  # endregion

  # region image preprocessing

  def adjust_image_quality(self, image_list, coeff_brightness=1.0, coeff_contrast=1.0, coeff_gamma=1.0):
    '''Adjust image quality using brightness, contrast and gamma setting.
    
    Args:
      image_list (list of 3D numpy arrays):
        Images to be adjusted
      coeff_brightness (float):
        Coefficient for brightness adjustment (default: 1.0)
      coeff_contrast (float):
        Coefficient for contrast adjustment (default: 1.0)
      coeff_gamma (float):
        Coefficient for gamma adjustment (default: 1.0)

    Returns:
      image_bcg_list (list of 3D numpy arrays):
        List of adjusted images

    Raises:
    '''
    image_bcg_list = []
    for image in image_list:

      enhancer_brightness = ImageEnhance.Brightness(image)
      image_b = enhancer_brightness.enhance(coeff_brightness)
      enhancer_contrast = ImageEnhance.Contrast(image_b)
      image_bc = enhancer_contrast.enhance(coeff_contrast)
      image_bcg = image_bc.point(lambda x: x**coeff_gamma)
      image_bcg_list.append(image_bcg)

    return image_bcg_list

  def make_image_size_equal(self, folder_source=None, folder_processed=None, name_file_1=None, name_file_2=None):
    '''This function takes images in a specified folder structures, mirrors the folder sturcture and traverses through the source folder while equliazing the image size and copying the results in the corresponding folder of the new structure.
     
    Args:
      folder_source (str):      Source folder (default: None)
      folder_processed (str):   Folder of processed files (default: None)
      name_file_1 (str):        Name of first file (default: None)
      name_file_2 (str):        Name of second file (default: None)
      
    Returns:
     
    Raises:
     '''
    
    # Walk through the directory structure
    for dir_0 in os.listdir(folder_source):
      
      if os.path.isdir(os.path.join(folder_source, dir_0)):
        source_dir_0 = os.path.join(folder_source, dir_0)
        target_dir_0 = os.path.join(folder_processed, dir_0)
        os.makedirs(target_dir_0, exist_ok=True)

        for dir_1 in os.listdir(source_dir_0):

          source_dir_1 = os.path.join(source_dir_0, dir_1)
          target_dir_1 = os.path.join(target_dir_0, dir_1)
          os.makedirs(target_dir_1, exist_ok=True)

          # Process images in the current subfolder
          image_source_paths = [os.path.join(source_dir_1, name_file_1), 
                                os.path.join(source_dir_1, name_file_2)]
          image_target_paths = [os.path.join(target_dir_1, name_file_1), 
                                os.path.join(target_dir_1, name_file_2)]
          
          if len(image_source_paths) == 2:  # Ensure there are exactly two images
              # Load the images
              img1 = cv2.imread(image_source_paths[0])
              img2 = cv2.imread(image_source_paths[1])
              
              if img1 is not None and img2 is not None:
                  # Get dimensions of both images
                  h1, w1 = img1.shape[:2]
                  h2, w2 = img2.shape[:2]

                  # Determine the new size based on the smaller image
                  new_height = min(h1, h2)
                  new_width = min(w1, w2)

                  # Resize both images
                  img1_resized = cv2.resize(img1, (new_width, new_height))
                  img2_resized = cv2.resize(img2, (new_width, new_height))

                  # Save the resized images in the target directory
                  cv2.imwrite(image_target_paths[0], img1_resized)
                  cv2.imwrite(image_target_paths[1], img2_resized)

  def calculate_common_similarity_metrics(self, folder_source=None, folder_processed=None, delta_list=[0.1, 0.1, 0.1, 0.1]):
    '''This function calculates common similarity metrics, e.g. SSIM, MSE, NMI, RE, VOI, RMSE, FSIM, PSNR, SAM, SRE, UIQ.
    
    Args:
      folder_source (str):
        Source folder of images (default: None)
      folder_processed (str):
        Processed folder (default: None)
      delta_list (list of float):
        Values for binarization threshold (default: [0.1, 0.1, 0.1, 0.1])

    Returns:
      folder_results_date (str): 
        Relative path of results folder since all folder name are expanded by a date and time string.

    Raises:
    
    '''
    
    Buffer = 3

    path_current = os.getcwd()
    path_folders = os.path.join(path_current, folder_source)
    Folders = []
    for item in os.listdir(path_folders):
        item_path = os.path.join(path_folders, item)
        if os.path.isdir(item_path):
            Folders.append(item)
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_processed = folder_processed + '_' + date_string
    Simi_Index = pd.DataFrame(columns=['cookie', 
                                      'dirt',
                                      'dirt_size',
                                      'paper',
                                      'cwssim',
                                      'ssim', 
                                      'mse', 
                                      'psnr',
                                      'are', 
                                      'voi',
                                      'fsim',
                                      'chi2',
                                      'mos'])

    # Get values of MOS metric
    df_mos = pd.read_csv(os.path.join(path_folders, 'MOS.txt'), 
                         header=None, 
                         names=['file_name', 'mos'])

    # Create main folder of results
    if os.path.isdir(folder_processed):
        print("The directory is already created.") 
    else:
        os.mkdir(folder_processed)
        print("The directory is created.")

    for i_sample_type in range(0, 4):
        
        delta = delta_list[i_sample_type]

        # Create result folders
        Folder_name = Folders[i_sample_type]
        images = os.listdir(os.path.join(path_folders, Folder_name))
        recognized_parts = os.path.join(path_current, 
                                        folder_processed + '\\' + Folder_name)
        if os.path.isdir(recognized_parts):
          print("The directory is already created.") 
        else:
          os.mkdir(recognized_parts)
          print("The directory is created.")
        
        # Apply calculation on each image pair
        for IImage in range(0, len(images)):
            
            # Get image paths
            image_name = images[IImage]
            path_THz = os.path.join(os.path.join(path_folders, Folder_name), image_name)
            THz_img = os.listdir(path_THz)

            # Load images and assign
            Img_THz_Ref = os.path.join(path_THz, THz_img[1])
            Img_THz_Cont = os.path.join(path_THz, THz_img[0])
            img_ref_original = sk.io.imread(Img_THz_Ref)
            img_cont_original = sk.io.imread(Img_THz_Cont)
            img_Ref = img_ref_original
            img_Cont = img_cont_original
            
            img_Ref = sk.io.imread(Img_THz_Ref)
            img_Ref = img_Ref[Buffer:(int(np.shape(img_Ref)[0]-Buffer)), Buffer:(int(np.shape(img_Ref)[1]-Buffer)), :]
            img_Cont = sk.io.imread(Img_THz_Cont)

            img_Cont = img_Cont[Buffer:(int(np.shape(img_Cont)[0]-Buffer)), Buffer:(int(np.shape(img_Cont)[1]-Buffer)), :]

            if img_Ref.shape[0] > img_Cont.shape[0]:
                img_Ref = img_Ref[0:int(np.shape(img_Cont)[0]), :, :]
            elif img_Cont.shape[0] > img_Ref.shape[0]:
                img_Cont = img_Cont[0:int(np.shape(img_Ref)[0]), :, :]
            
            # Process reference state
            hsv_img = sk.color.rgb2hsv(img_Ref)
            hue_img = hsv_img[:, :, 0]
            sat_img = hsv_img[:, :, 1]
            value_img = hsv_img[:, :, 2]
            processed_Channel = hue_img
            Thershold = (np.mean(processed_Channel)+(delta*np.std(processed_Channel)))*100
            for I in range(0, np.shape(processed_Channel)[0]):
                for K in range(0, np.shape(processed_Channel)[1]):
                    if int(processed_Channel[I][K]*100) <= Thershold:
                        processed_Channel[I][K] = 0
                    else:
                        processed_Channel[I][K] = 1
            binary_1 = np.array(processed_Channel, dtype=bool)
            binary_1 = sk.morphology.remove_small_objects(binary_1, 10)
            
            labeled_image_1 = sk.measure.label(binary_1)
            object_features = sk.measure.regionprops(labeled_image_1)
            regions_1 = sk.measure.regionprops_table(labeled_image_1, properties=(['area']))
            Features_1 = pd.DataFrame(regions_1)
            for object_id, objf in enumerate(object_features, start=1):
                if objf["area"] > 100:
                    labeled_image_1[labeled_image_1 == objf["label"]] = 0
            ref_binary = np.array(labeled_image_1, dtype=bool)
            
            # Process contaminated state
            hsv_img = sk.color.rgb2hsv(img_Cont)
            hue_img = hsv_img[:, :, 0]
            sat_img = hsv_img[:, :, 1]
            value_img = hsv_img[:, :, 2]
            processed_Channel = hue_img
            Thershold = (np.mean(hue_img)+(delta*np.std(hue_img)))*100
            for I in range(0, np.shape(processed_Channel)[0]):
                for K in range(0, np.shape(processed_Channel)[1]):
                    if int(processed_Channel[I][K]*100) <= Thershold:
                        processed_Channel[I][K] = 0
                    else:
                        processed_Channel[I][K] = 1
            
            binary_2 = np.array(processed_Channel, dtype=bool)
            binary_2 = sk.morphology.remove_small_objects(binary_2, 10)
            
            labeled_image_2 = sk.measure.label(binary_2)
            object_features = sk.measure.regionprops(labeled_image_2)
            regions_2 = sk.measure.regionprops_table(labeled_image_2, properties=(['area']))
            Features_2 = pd.DataFrame(regions_2)
            for object_id, objf in enumerate(object_features, start=1):
                if objf["area"] > 100:
                    labeled_image_2[labeled_image_2 == objf["label"]] = 0
            cont_binary = np.array(labeled_image_2, dtype=bool)
            
            # Calculate metrics using toolbox 'skikit-image', e.g. MSE, ARE, ref. NMI, VOI
            (ssim, diff) = structural_similarity(ref_binary, cont_binary, full=True)
            mse = mean_squared_error(ref_binary, cont_binary)
            are = adapted_rand_error(ref_binary, cont_binary)
            if math.isnan(are[0]):
               are_result = 0.0
            else:
               are_result = are[0]
            nmi = normalized_mutual_information(ref_binary, cont_binary)
            ref_nmi = normalized_mutual_information(ref_binary, ref_binary)
            voi = variation_of_information(ref_binary, cont_binary)
            #HD = hausdorff_distance(ref_binary, cont_binary)
            #HP = hausdorff_pair(ref_binary, cont_binary)
            #Ref_NRMSE = normalized_root_mse(ref_binary, ref_binary)
            #NRMSE = normalized_root_mse(ref_binary, cont_binary)
            #PSNR = peak_signal_noise_ratio(ref_binary, cont_binary)
            
            # Apply CW‑SSIM metric calculation
            ssim_temp = SSIM(Img_THz_Ref, gaussian_kernel_1d=None)
            cwssim = ssim_temp.cw_ssim_value(Img_THz_Cont)
            #cwssim = self.calculate_cw_ssim(img_ref_original, img_cont_original)

            # Calculate additional similarity metrics using the toolbox 'image_similarity measures', e.g. RMSE, PSNR, SSIM, FSIM, ISSM, SRE, SAM, and UIQ
            metrics = evaluation(org_img_path=Img_THz_Ref, 
                                pred_img_path=Img_THz_Cont, 
                                metrics=["rmse", "psnr", "fsim", "sam", "sre", "ssim", "uiq"])

            # Apply Chi‑Square distance metric calculation to grayscale images
            top_gray = sk.color.rgb2gray(img_ref_original)
            bot_gray = sk.color.rgb2gray(img_cont_original)
            h1, b1 = np.histogram(top_gray, bins=256, range=(0, 1))
            h2, b2 = np.histogram(bot_gray, bins=256, range=(0, 1))
            h1_norm = h1 / (h1.sum() + 1e-10)
            h2_norm = h2 / (h2.sum() + 1e-10)
            chi2 = 0.5 * np.sum((h1_norm - h2_norm)**2 / (h1_norm + h2_norm + 1e-10))

            # Find value of VI metric and save to dataframe
            mos_entry = df_mos[df_mos['file_name'] == image_name]
            mos = mos_entry.iloc[0]['mos']

            # Get XOR gate difference image after CCA
            Diff = np.bitwise_xor(ref_binary, cont_binary)
            binary_3 = np.array(Diff, dtype=bool)
            binary_3 = sk.morphology.remove_small_objects(binary_3, 10)
            labeled_image_3 = sk.measure.label(binary_3)
            object_features = sk.measure.regionprops(labeled_image_3)
            regions_3 = sk.measure.regionprops_table(labeled_image_3, properties=('area', 'perimeter', 'axis_major_length', 'axis_minor_length'))
            Features_3 = pd.DataFrame(regions_3)
            for object_id, objf in enumerate(object_features, start=1):
                if objf["area"] > 60: #7000
                    labeled_image_3[labeled_image_3 == objf["label"]] = 0
                if objf["area"] < 35: #7000
                    labeled_image_3[labeled_image_3 == objf["label"]] = 0
                if objf["axis_minor_length"] < 5.11: #7000
                    labeled_image_3[labeled_image_3 == objf["label"]] = 0
                if objf["axis_major_length"] > 18: #7000
                    labeled_image_3[labeled_image_3 == objf["label"]] = 0
                if objf["perimeter"] > 34: #7000
                    labeled_image_3[labeled_image_3 == objf["label"]] = 0
            Diff_CCA = np.array(labeled_image_3, dtype=bool)
            
            # Draw XOR gate difference image after CCA
            plt.figure()
            plt.imshow(Diff_CCA, cmap=plt.cm.gray) 
            plt.axis('off')

            # Save meta data and results to a dataframe
            parts_name = image_name.split('_')
            Simi_Index.loc[IImage, 'cookie'] = parts_name[0]
            Simi_Index.loc[IImage, 'dirt'] = parts_name[1]
            Simi_Index.loc[IImage, 'dirt_size'] = parts_name[2][0]
            Simi_Index.loc[IImage, 'paper'] = parts_name[3][0:-1]
            Simi_Index.loc[IImage, 'ssim'] = ssim
            Simi_Index.loc[IImage, 'cwssim'] = cwssim
            Simi_Index.loc[IImage, 'mse']  = mse
            Simi_Index.loc[IImage, 'psnr'] = metrics['psnr']
            Simi_Index.loc[IImage, 'are']  = are_result
            Simi_Index.loc[IImage, 'voi']  = voi[0]
            Simi_Index.loc[IImage, 'fsim'] = metrics['fsim']
            Simi_Index.loc[IImage, 'chi2'] = chi2
            Simi_Index.loc[IImage, 'mos'] = mos

            # Save the dataframe to CSV file
            save_path = (folder_processed + "\\similarity_indices_" + 
                         Folder_name + '.csv')
            Simi_Index.to_csv(save_path, index=False)

            # Save the diagram as an image
            plt.savefig(os.path.join(recognized_parts, image_name +'.jpg'))
            plt.close()

    return folder_processed

  def calculate_cw_ssim(self, image_1=None, image_2=None, K=0.03):
    '''Calculate the CW-SSIM value of two images that are compared.
    
    Args:
      image_1 (numpy array):    
        First image (default: None)
      image_2 (numpy array):    
        Second image (default: None)
      K (float):                
        Small constant (default: 0.03)

    Returns:
      cw_ssim (float):        
        CW-SSIM value

    Raises:
    '''

    # Perform complex wavelet transform on both images
    coeffs_1 = pywt.wavedec2(image_1, 'bior1.3', level=1)
    coeffs_2= pywt.wavedec2(image_2, 'bior1.3', level=1)

    # Extract the detail coefficients
    cA_A, (cH_A, cV_A, cD_A) = coeffs_1
    cA_B, (cH_B, cV_B, cD_B) = coeffs_2

    # Calculate the CW-SSIM
    num = np.sum(cA_A * cA_B) + np.sum(cH_A * cH_B) + np.sum(cV_A * cV_B) + np.sum(cD_A * cD_B) + K
    denom = np.sqrt(np.sum(cA_A ** 2) * np.sum(cA_B ** 2)) + np.sqrt(np.sum(cH_A ** 2) * np.sum(cH_B ** 2)) + \
            np.sqrt(np.sum(cV_A ** 2) * np.sum(cV_B ** 2)) + np.sqrt(np.sum(cD_A ** 2) * np.sum(cD_B ** 2)) + K

    cw_ssim = num / denom

    return cw_ssim

  def calculate_similarity_index(self, image_path_reference=None, 
                                 image_path_sample=None,
                                 mode=None):
    '''Calculate similarity index.
    
    Args:
      image_path_reference (str): 
        Path to reference image (default: None)
      image_path_sample (str): 
        Path to sample image (default: None)
      mode:
        Mode of similarity index calculation (default: None)

    Returns:
      folder_results_date (numpy array):


    Raises:
    '''
    # Load images
    img_ref = cv2.imread(image_path_reference)
    img_spl = cv2.imread(image_path_sample)

    # Plot images
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
    ax[0].imshow(img_gray_ref)
    ax[0].set_title('Reference')
    ax[1].imshow(img_gray_spl)
    ax[1].set_title('Sample')
    plt.tight_layout()
    plt.show()

    # Convert to grayscale
    img_gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    img_gray_spl = cv2.cvtColor(img_spl, cv2.COLOR_BGR2GRAY)

    folder_results_date = None

    return folder_results_date

  def calculate_statistics_of_similarity(self, csv_file_path=None, parms=None, save_stats_to_tex=False):
    '''Calculate statistics of similarity.
    
    Args:
      csv_file_path (string): 
        Relative path of CSV files (default: None)
      parms (dict):
        Parameters (default: None)
      save_stats_to_tex (bool):
        Save statistics to TeX file (default: False)

    Returns:

    Raises:
    
    '''

    FACTORS = parms['factors']
    NAMES_FACTORS = parms['names_factors']
    VARIABLES = parms['variables']
    NAMES_VARIABLES = parms['names_variables']
    PRINT_HEAD_TAIL = parms['print_head_tail']

    # region Computation
    #-------------------
    # List to hold CSV file names
    csv_files = []
    for filename in os.listdir(csv_file_path):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(csv_file_path, filename))

    # Read CSV file and convert to data frame
    if len(csv_files) > 1:
      df_list = []
      for i in range(len(csv_files)):
        df_i = pd.read_csv(csv_files[i])
        df_list.append(df_i)
      df = pd.concat(df_list, ignore_index=True)
    else:
      df = pd.read_csv(csv_files)

    # Cast factors as categories
    for col in FACTORS.keys():
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Show some results
    if PRINT_HEAD_TAIL:
        print(df.head())
        print(df.tail())

    # Filter data frame
    df_group = {}
    for key in FACTORS.keys():
        df_subgroup = {}
        for value in df[key]:
            df_subgroup[value] = df[df[key] == value]
        df_group[key] = df_subgroup

    # Calculate mean, median and variance for every subgroup
    mean_group = {}
    median_group = {}
    var_group = {}
    for key in FACTORS.keys():
        mean_subgroup = {}
        median_subgroup = {}
        var_subgroup = {}
        for value in df[key]:
            mean_subgroup[value] = df_group[key][value][VARIABLES[0]].mean()
            median_subgroup[value] = df_group[key][value][VARIABLES[0]].median()
            var_subgroup[value] = df_group[key][value][VARIABLES[0]].var()
        mean_group[key] = mean_subgroup
        median_group[key] = median_subgroup
        var_group[key] = var_subgroup

    # Save results to TeX file
    if save_stats_to_tex:

      names_column = ['Item', 'Mean', 'Median', 'Variance']
      category = list(mean_group.keys())

      df_stats = {key: None for key in category}
      for i in range(len(df_stats.keys())):
        elem = list(mean_group[category[i]].keys())
        df_i = pd.DataFrame(columns=names_column, index=elem)
        for j in elem:
          df_i.loc[j, names_column[0]] = j
          df_i.loc[j, names_column[1]] = mean_group[category[i]][j]
          df_i.loc[j, names_column[2]] = median_group[category[i]][j]
          df_i.loc[j, names_column[3]] = var_group[category[i]][j]
        df_stats[category[i]] = df_i

      with open(os.path.join(csv_file_path, 'statistics.tex'), 'w') as f:
        for i in range(len(df_stats)):

          f.write(df_stats[category[i]].to_latex(index=False, 
                                                 caption=f'Statistics of {category[i]}', 
                                                 label=f'tab:statistics_{category[i]}'))
          f.write('\n\n')

    # Calculate detection performance based on rule-based method
    thd = np.arange(0.0, 1.0, 0.1)
    tpr = []
    for thd_i in thd:
      df_tp = df[df['ssim'] < thd_i]
      tpr_i = df_tp.shape[0]/df.shape[0]
      tpr.append(tpr_i)

    # Plot mean, median and variance for every subgroup
    def plot_boxplots(key=None, n=0, save=False, dim=None):

        if dim is None:
          fig, ax = plt.subplots(nrows=2, ncols=int(np.ceil(n/2)), figsize=(14, 8))
        else:
           fig, ax = plt.subplots(nrows=dim[0], ncols=dim[1], figsize=(14, 8))
        fig.canvas.manager.set_window_title(NAMES_FACTORS[key])
        ax = ax.flatten()
        for i, value in enumerate(FACTORS[key]):
            
            ax[i].set_title(NAMES_FACTORS[value])
            x = df_group[key][value][VARIABLES]

            # Normalize
            x['mos'] = np.interp(x['mos'], [1, 5], [0.0, 1.0])
            x['psnr'] = x['psnr'] / 100.0

            ax[i].boxplot(x, vert=True, patch_artist=True)
            ax[i].set_ylabel('Values')
            ax[i].set_xticks([j for j in range(1, len(NAMES_VARIABLES)+1)],
                             NAMES_VARIABLES)
            ax[i].set_ylim(0, 1)
        plt.tight_layout()
        #plt.show()
        if save:
            plt.savefig(f'{csv_file_path}\\statistics_{key}.jpg', format='jpg')
        plt.close()

    plot_boxplots(key='cookie', n=len(FACTORS['cookie']), save=True)
    plot_boxplots(key='dirt', n=len(FACTORS['dirt']), save=True)
    plot_boxplots(key='dirt_size', n=len(FACTORS['dirt_size']), save=True)
    plot_boxplots(key='paper', n=len(FACTORS['paper']), save=True, dim=[3, 1])

    print("Computation completed.")

  # endregion

  def calculate_anova(self, csv_file_path=None):
    '''Apply 3-way annova.
    
    Args:
      csv_file_path (str):    File path (default: None)

    Returns:

    Raises:
    '''

    # Define output file
    output_file_path = os.path.join(csv_file_path, 'annova')
    os.makedirs(output_file_path, exist_ok=True)

    # List to hold CSV file names
    csv_files = []
    for filename in os.listdir(csv_file_path):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(csv_file_path, filename))

    # Read CSV file and convert to data frame
    if len(csv_files) > 1:
      df_list = []
      for i in range(len(csv_files)):
        df_i = pd.read_csv(csv_files[i])
        df_list.append(df_i)
      df = pd.concat(df_list, ignore_index=True)
    else:
      df = pd.read_csv(csv_files)

     # Cast factors as categories
    for col in ["cookie", "dirt", "dirt_size", "paper"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Harmonize metrics if required (temporary)
    df["ssim_vis"] = df["ssim"]
    df["cwssim_vis"] = df["cwssim"]
    df["mse_vis"] = df["mse"]

    # Define metrics
    metrics_to_run = {"ssim_vis": "SSIM",
                      "cwssim_vis": "CWSSIM",
                      "mse_vis": "MSE",
                      "mos" : "mos"}

    # Define formula
    formula_3way = ("value ~ C(cookie)*C(dirt)*C(dirt_size)"
                    " + C(cookie)*C(dirt)*C(paper)"
                    " + C(cookie)*C(dirt_size)*C(paper)"
                    " + C(dirt)*C(dirt_size)*C(paper)")

    all_results = {}

    anova_tbl_all = []
    for mcol, mlabel in metrics_to_run.items():
        d = df.copy()
        d = d.rename(columns={mcol: "value"}).dropna(subset=["value"])
        #d = d.rename(columns={mcol: "value"})

        model = ols(formula_3way, data=d).fit()
        anova_tbl = anova_lm(model, typ=2)

        all_results[mcol] = {"label": mlabel, "model": model, "anova": anova_tbl}

        anova_path = os.path.join(output_file_path, f"anova_{mcol}.csv")
        anova_tbl.to_csv(anova_path, index=True)
        anova_tbl_all.append(anova_tbl)

    # Save to a .tex file
    with open(os.path.join(csv_file_path, 'anova.tex'), 'w') as f:
        for i, tex_i in enumerate(anova_tbl_all):
          f.write(tex_i.to_latex(index=True))
          f.write("\n")

  # region CNN

  def cnn_encoder(self, input_dim=(256, 256, 3), plot_summary=False):
    '''Set up CNN-based encoder, e.g. for feature extraction.
    
    Args:
      input_dim (int):
        Input dimension (default: (256, 256, 3))
      plot_summary (bool):
        Plot summary (default: False)

    Returns:

    Raises:
    '''

    cnn_encoder = None
    cnn_encoder_output = None
    cnn_encoder_input = Input(shape=input_dim, dtype=np.float32)

    num_conv = {0: {'filters': 32, 'kernel_size': 3, 'strides': 1},
                  1: {'filters': 64, 'kernel_size': 3, 'strides': 1},
                  2: {'filters': 128, 'kernel_size': 3, 'strides': 1},
                  3: {'filters': 265, 'kernel_size': 3, 'strides': 1},
                  4: {'filters': 512, 'kernel_size': 3, 'strides': 1}}

    cnn_encoder_output = cnn_encoder_input
    for layer_num, layer_data in num_conv.items():
        cnn_encoder_output = Conv2D(layer_data['filters'], 
                                    layer_data['kernel_size'], 
                                    layer_data['strides'], 
                                    padding='same')(cnn_encoder_output)
        cnn_encoder_output = MaxPooling2D((2,2), 
                                          padding='same')(cnn_encoder_output)
        cnn_encoder_output = LeakyReLU(alpha=0.2)(cnn_encoder_output)
        cnn_encoder_output = BatchNormalization(axis=-1)(cnn_encoder_output)

    encoder = Model(cnn_encoder_input, cnn_encoder_output)

    if plot_summary:
      encoder.summary()

    return encoder

  # region CAE

  def set_cae_encoder(self, input_dim, z_dim):
    '''This function defines the encoder part of the convolutional autoencoder (CAE)
    
    Args:
      input_dim (tupel):
        Size of input layer (rows, columns, channels)
      z_dim (int):
        Size of latent vector

    Returns:
      cae_encoder_model (Model):
        Encoder model

    Raises:
    '''

    cae_encoder = None
    cae_encoder_output = None
    cae_encoder_input = Input(shape = input_dim, dtype=np.float32)

    num_conv = {0: {'filters': 32, 'kernel_size': 3, 'strides': 1},
                  1: {'filters': 64, 'kernel_size': 3, 'strides': 1},
                  2: {'filters': 128, 'kernel_size': 3, 'strides': 1},
                  3: {'filters': 265, 'kernel_size': 3, 'strides': 1},
                  4: {'filters': 512, 'kernel_size': 3, 'strides': 1}
                  }

    encoder_2 = cae_encoder_input
    for layer_num, layer_data in num_conv.items():
        encoder_2 = Conv2D(layer_data['filters'], layer_data['kernel_size'], layer_data['strides'], padding='same')(encoder_2)
        encoder_2 = MaxPooling2D((2,2),padding='same')(encoder_2)
        encoder_2 = LeakyReLU(alpha=0.2)(encoder_2)
        encoder_2 = BatchNormalization(axis=-1)(encoder_2)

    int_shape = K.int_shape(encoder_2)
    flatten_2 = Flatten()(encoder_2)
    cae_encoder_output = Dense(z_dim)(flatten_2)
    cae_encoder = Model(cae_encoder_input, cae_encoder_output)

    cae_encoder.summary()

    return cae_encoder, cae_encoder_input, int_shape, num_conv

  def set_cae_decoder(self, int_shape, num_conv, n_channel=3, z_dim=512):
    '''This function defines the decoder part of the convolutional autoencoder (CAE)
    
    Args:
      int_shape (tupel):
        Size
      num_conv (dict):
        Convolutional layer parameters (filters, kernel size, stride)
      n_channel (int):
        Number of channels (default: 3)
      z_dim (int):
        Size of latent vector (default: 512)

    Returns:

    Raises:
    '''

    cae_decoder = None
    cae_decoder_output = None
    cae_decode_input = Input(shape=(z_dim,))
    decoder = Dense(np.prod(int_shape[1:]))(cae_decode_input)
    decoder = Reshape((int_shape[1], int_shape[2], int_shape[3]))(decoder)

    for layer_num,layer_data in reversed(sorted(num_conv.items())):
        decoder = Conv2DTranspose(layer_data['filters'], layer_data['kernel_size'], layer_data['strides'], padding='same')(decoder)
        #if layer_num < 2:
        decoder = UpSampling2D(size=(2,2))(decoder)
        decoder = LeakyReLU(alpha=0.2)(decoder)
        decoder = BatchNormalization(axis=-1)(decoder)

    decoder = Conv2DTranspose(n_channel, 3, padding='same')(decoder)
    cae_decoder_output_2 = Activation('sigmoid')(decoder)
    cae_decoder_2 = Model(cae_decode_input, cae_decoder_output_2)

    cae_decoder_2.summary()

    return cae_decoder_2

  def set_cae_model(self, cae_encoder_input, input_dim=(256, 256, 3), z_dim=512, type_loss='mse'):
    '''This function defines the CAE model consisting of the encoder and decoder part.
    
    Args:
      cae_encoder_input (Dense):
        CAE encoder input layer
      input_dim (tupel):
        Size of input layer (rows, columns, channels) (default: 256, 256, 3)
      z_dim (int):
        Size of latent vector (default: 512)
      type_loss (str):
        Type of loss: 'mse' (default), 'ssim_loss'

    Returns:
      cae_model (Model):
        CAE model

    Raises:
    '''
    (cae_encoder, 
     cae_encoder_input, 
     int_shape, 
     num_conv) = self.set_cae_encoder(input_dim, z_dim)
    cae_decoder = self.set_cae_decoder(self, int_shape, num_conv, 
                                       n_channel=3, z_dim=z_dim)

    cae_model = Model(cae_encoder_input, 
                      cae_decoder(cae_encoder(cae_encoder_input)))
    
    if type_loss == 'mse':
      cae_model.compile(optimizer='adam', loss='mse')
    elif type_loss == 'ssim_loss':
      cae_model.compile(optimizer='adam', loss=self.ssim_loss)

    cae_model.summary()
    cae_model.get_config()

    return cae_model

  # endregion

  # region GAN

  def set_gan_generator(self, image_size, latent_code_length):
    '''Set the generator part of the GAN.'''

    x = Input(latent_code_length)
    y = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(x)
    y = LeakyReLU()(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(128, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(64, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(image_size[-1],(3,3),strides=(2,2),padding="same")(y)

    return Model(x, y)

  def set_gan_encoder(image_size, latent_code_length):
    '''Set GAN encoder'''

    x = Input(image_size)
    y = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(x)
    y = LeakyReLU()(y)
    y = Conv2D(64, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(128, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(512, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(latent_code_length[-1],(3,3),strides=(2,2),padding="same")(y)

    return Model(x, y)

  def set_gan_discriminator(self, image_size, latent_code_length):
    '''Set the discriminator part of the GAN.'''

    x = Input(image_size)
    z = Input(latent_code_length)
    _z = Flatten()(z)
    _z = Dense(image_size[0]*image_size[1]*image_size[2])(_z)
    _z = Reshape(image_size)(_z)

    y = Concatenate()([x,_z])
    y = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(128, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(512, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(1024, (3, 3), strides=(2, 2), padding="same")(y)
    y = LeakyReLU()(y)
    y = Conv2D(1024, (3, 3), padding="same")(y)
    y = LeakyReLU()(y)
    y = Flatten()(y)
    y = Dense(1)(y)

    return Model([x, z], [y])

  def build_train_step(self, generator, encoder, discriminator):
    '''Build train step procedure.'''
    pass

  def train_gan(self, x_train):
    '''Perform training of the GAN.'''

  def ssim_loss(y_true, y_pred):
      '''Calculate SSIM loss.
      
      Args:
        y_true (float, image, or list of these datatypes):
          True ground image
        y_pred (float, image, or list of these datatypes):
          Predcited image

      Returns:
        ssim_loss (float):
          Loss function value
      
      '''
      return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

  # endregion

  # region own anomaly detection models, training, and evaluation

  def create_base_network(self, type_feature_module='ViT_DeiT', 
                 input_dim=(256, 256, 3), type_fusion_module='Dense', plot_summary=False, set_mha=True, n_patches_mha=16):
    '''Create base network required for dual instantiation in order to apply anomaly detection.
    
    Args:
      type_feature_module (str):
        Type of feature extractor:
          'CNN': Convolutional Neural Networks with MaxPooling-, LeakyReLU- and BatchNormalization-Layers
          'ViT_DeiT': Data-efficient Image Transformer (https://paperswithcode.com/method/deit)          -> TODO
          'ViT_MobileViT':          -> TODO

          'ViT_CrossViT':           -> TODO

          'ViT_Swin-Transformer':   -> TODO

      input_dim (int):
        Input dimension (default: (256, 256, 3))
      plot_summary (bool):
        Plot summary (default: False)
      set_mha (bool):
        Turn on/off MHA module (default: True)
      n_patches_mha (int):
        Number of patches of MHA (default: 16)

    Returns:

    Raises:
    '''
  
    # Set input for the overall model
    inputs = Input(shape=input_dim)

    # Select feature extractor (feature module)
    if type == 'CNN':
      
      feature_module = self.cnn_encoder(self, input_dim, plot_summary)

    elif type == 'ViT_DeiT':
      pass
    elif type == 'ViT_MobileViT':
      pass
    elif type == 'ViT_CrossViT':
      pass
    elif type == 'ViT_Swin-Transformer':
      pass

    # Get feature module
    feature_module_output = feature_module(inputs)

    # Set Multi-Head Attention (MHA) network if desired with skip connection
    if set_mha:

      # Flatten input and pass to MHA module
      outputs = Flatten()()
      features = feature_module_output.output_shape[-1] / n_patches_mha
      outputs = Reshape((n_patches_mha, features))(outputs)
      attention_output = MultiHeadAttention(num_heads=4, 
                                            key_dim=features)(outputs, outputs)

      # Add skip connection
      outputs = Add()([feature_module_output, attention_output])
    else:
      outputs = feature_module_output

    # Set fusion mechanism
    if type_fusion_module == 'Dense':
      outputs = Flatten()(outputs)
      outputs = Dense(128, activation='relu')(outputs)

    return Model(inputs, outputs)

  def create_snm2ad(self, type_feature_module='ViT_DeiT', 
                 input_dim=(256, 256, 3), type_fusion_module='Dense', plot_summary=False, set_mha=True, n_patches_mha=16, margin=1):
    '''Define own multi-modal anomaly detection method called Siamese-Network-Based Multi-Modal Anomaly Detection Model (SNM2AD).
    
    Args:
      type_feature_module (str):
        Type of feature extractor:
          'CNN': Convolutional Neural Networks with MaxPooling-, LeakyReLU- and BatchNormalization-Layers
          'ViT_DeiT': Data-efficient Image Transformer (https://paperswithcode.com/method/deit)
          'ViT_MobileViT':

          'ViT_CrossViT':

          'ViT_Swin-Transformer':

      input_dim (int):
        Input dimension (default: (256, 256, 3))
      plot_summary (bool):
        Plot summary (default: False)
      set_mha (bool):
        Turn on/off MHA module (default: True)
      n_patches_mha (int):
        Number of patches of MHA (default: 16)
      margin (int):
        Margin (default: 1)

    Returns:

    Raises:
    '''

    # Create base models
    base_model = self.create_base_network(type_feature_module,
                                              input_dim, 
                                              type_fusion_module, 
                                              plot_summary, 
                                              set_mha, 
                                              n_patches_mha)

    # Define inputs for the Siamese network
    input_1 = Input(shape=input_dim)
    input_2 = Input(shape=input_dim)

    # Get outputs from the CNN
    output_1 = base_model(input_1)
    output_2 = base_model(input_2)

    # Implement loss function
    merge_layer = keras.layers.Lambda(self.euclidean_distance, 
                                      output_shape=(1,))([output_1, 
                                                          output_2])
    normal_layer = keras.layers.BatchNormalization()(merge_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
    
    # Set final model
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    # Compile model
    siamese.compile(loss=self.siamese_network_loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])
    siamese.summary()


    # history = siamese.fit(
    #     [x_train_1, x_train_2],
    #     labels_train,
    #     validation_data=([x_val_1, x_val_2], labels_val),
    #     batch_size=batch_size,
    #     epochs=epochs,
    # )

  def euclidean_distance(vects):
      '''Find the Euclidean distance between two vectors.

      Args:
          vects: List containing two tensors of same length.

      Returns:
          Tensor containing euclidean distance
          (as floating point value) between vectors.

      Raises
      '''
      x, y = vects
      sum_square = tf.ops.sum(tf.ops.square(x - y), axis=1, keepdims=True)
      return tf.ops.sqrt(tf.ops.maximum(sum_square, keras.backend.epsilon()))

  def siamese_network_loss(margin=1):
      '''Provides contrastive loss an enclosing scope with margin.

      Arguments:
          margin: Integer, defines the baseline for distance for which pairs
                  should be classified as dissimilar. - (default is 1).

      Returns:
          contrastive_loss: function with data ('margin') attached.

      Raises:
      '''

      def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = tf.ops.square(y_pred)
        margin_square = tf.ops.square(tf.ops.maximum(margin - (y_pred), 0))
        return tf.ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)
        # square_pred = tf.square(y_pred)
        # margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        # return tf.reduce_mean((1 - y_true) * square_pred + y_true * margin_square)

      return contrastive_loss

  def triplet_loss(y_true, y_pred, alpha=0.2):
    '''Calculate Triplet loss function.

    Parms:
      y_true: 
        True labels (not used in the loss calculation).
      y_pred: 
        A tensor of shape (batch_size, 3) containing the embeddings for
        the anchor, positive, and negative samples.
      alpha: Margin between positive and negative pairs.

    Returns:
      Triplet loss value.

    Raises:
    '''
    anchor, positive, negative = tf.unstack(y_pred, axis=1)

    # Compute the distances
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # Compute the triplet loss
    loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + alpha, 0.0))
    return loss

  # endregion

  # region Auxiliary methods

  def matlab_mmwstudio_startup(self, exe_path):
    """Apply startup of MATLAB and TI's mmWave Studio.
    
    Args:
      exe_path (str):
        File path of mmwstudio

    Returns:
      me (specific object): 
        MATLAB engine instance

    Raises:
    """
    
    # Send LUA script for initial configuration using a MATLAB script (workaround)
    me = matlab.engine.start_matlab()
    me.cd(os.getcwd(), nargout=0)
    _ = me.configmmwstudio()

    return me

  def configure_udp_socket(self, parms):
    """This functions configures the UDP socket, which is required for the communication with the measurement trigger.
    
    Args:
      parms (dict):
        Parameters

    Returns:
      sock (socket object):
        Socket for UDP communication

    Raises:
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((parms['udp_ip'], parms['udp_port']))
    sock.setblocking(False)

    return sock

  def listen_for_udp_trigger(self, sock, parms):
    """This functions listens for UDP-based trigger and sets a flag, if a valid message was received.
    
    Args:
    sock (socet object): 
      Socket for UDP communication
    parms (dict):
      Parameters

    Returns:
      trigger (bool):
        Trigger flag (True: Trigger on; False: Trigger off)
      parms (dict):
        Parameters
      timestamp_sender (str):
        Timestamp of sender
      timestamp_receiver (str):
        Timestamp of receiver

    Raises:
    """

    self.t_0 = None
    while True:
      try:
        data, addr = sock.recvfrom(parms['udp_max_n_byte'])
        message = data.decode('utf-8')
        print("-----------------")
        print("UDP communication")
        print("-----------------")
        print(f"UDP communication: Received UDP message from {addr} <{message}>")
        command, _ = message.split(',',1)

        if (command == 'trigger'):
          self.t_0 = time.time()
          print("Trigger detected!")
          break
      except (BlockingIOError, RuntimeError, ValueError):
        pass

  def save_results(self, results):
    """Save results to pickled data and to images in a new folder.
    
    Args:
      results (list):
        List of results: img_thz, data_thz, img_mv

    Returns:

    Raises:
    """

    if results is not None:
      
      # Dump results to folder
      current_time = datetime.now()
      formatted_time = current_time.strftime("%Y_%m_%d-%H_%M_%S")
      filename_prefix = f"ifi_"
      folder_name = f"measurements_ifi\\ifi_{formatted_time}"
      folder_path = os.path.join(os.getcwd(), folder_name)

      if not isinstance(results, list):
        results = [results]
      os.makedirs(folder_path, exist_ok=True)
      for i, f in enumerate(results):
        if isinstance(f, np.ndarray):
          plt.imsave(os.path.join(folder_path, filename_prefix + self.MEASUREMENT_TECHNOLOGIES[i] + '_' + formatted_time + '.png'), f.astype(np.uint8))

          # with open(os.path.join(folder_path, filename_prefix + self.MEASUREMENT_TECHNOLOGIES[i] + '_' + formatted_time + '.pkl'), 'wb')
          with open(os.path.join(folder_path, filename_prefix + self.MEASUREMENT_TECHNOLOGIES[i] + '_' + formatted_time + '.npy'), 'wb') as file:
            #pickle.dump(f, file)
            np.save(file, f)

      results = [None] * 3
      print(f"Files saved to ({folder_path}).")

  def peek_results_pickled(self, folder_path):
    """Peek into results from pickled data.
    
    Args:
      folder_path (list):
        Path to folder containing pickle files (THz image, MV image)

    Returns:
      data (list):
        List of data (THz image, MV image)

    Raises:
    """
    pickled_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pkl'):
                pickled_files.append(os.path.join(root, file))

    data = [i for i in range(len(pickled_files))]
    for i in range(len(pickled_files)):
      with open(pickled_files[i], 'rb') as file:
        data[i] = pickle.load(file)

        # Plot the image
        plt.figure()
        plt.imshow(data[i])
        path_splits = pickled_files[i].split('\\')
        file_splits = path_splits[2].split('_')
        plt.title(file_splits[1].upper())
        plt.show(block=False)

  def peek_results_input(self, input_data, group='anomalous', idx=0):
    """Peek into results already loaded as input inside of the analysis function.
    
    Args:
      input_data (list):
        Anomalous and normal data arranged as lists, where each list containes instances. Each instance has the THz, the FMCW and MV image.
      group (str):
        Selector for group of instances (default: 'anomalous')
      idx (int):
        Instance index (default: 0)

    Returns:
      data (list):
        List of data (THz image, MV image)

    Raises:
    """
    _, ax = plt.subplots(nrows=1, ncols=3)
    if group == 'normal':
      idx_type = 0
    elif group == 'anomalous':
      idx_type = 1
    ax[0].imshow(input_data[idx_type][idx][0])
    ax[1].imshow(input_data[idx_type][idx][1])
    ax[2].imshow(input_data[idx_type][idx][2])
    ax[0].set(title='CV/MV')
    ax[1].set(title='THz')
    ax[2].set(title='THz (raw)')
    plt.tight_layout()
    plt.show(block=False)

  def convert_csv_files_to_tex_table(self, csv_file_folder, export_file_name):
    '''This function reads type-specific CSV files, concatenates the dataframes inot one big dataframe after reading, which is converted into TeX code and saved as a tex file.
    
    Args:
      csv_file_folder (list of strings):  File paths of type-specific CSV files
      export_file_name (string):          Tex file name for saving the  
                                          converted data

    Returns:

    Raises:
    '''

    # List to hold CSV file names
    csv_files = [f for f in os.listdir(csv_file_folder) if f.endswith('.csv')]

    # Read data frames
    df = [pd.read_csv(os.path.join(csv_file_folder, f)) for f in csv_files]

    # Split dataframe
    df_left = []
    for df_i in df:
      df_left_i = df_i.iloc[:, 0:8]
      df_left_i.rename(columns={'cookie': 'Type', 
                              'dirt': 'FO', 
                              'dirt_size': 'Size', 
                              'paper': 'Atten.',
                              'cwssim': 'CWSSIM',
                              'ssim': 'SSIM',
                              'mse': 'MSE',
                              'psnr': 'PSNR'}, inplace=True)
      df_left.append(df_left_i)

    df_right = []
    for df_i in df:
      first_four_columns = df_i.iloc[:, :4]
      last_five_columns = df_i.iloc[:, -5:]
      df_right_i = pd.concat([first_four_columns, 
                                 last_five_columns], axis=1)
      df_right_i.rename(columns={'cookie': 'Type', 
                              'dirt': 'FO', 
                              'dirt_size': 'Size', 
                              'paper': 'Atten.',
                              'are': 'ARE',
                              'voi': 'VoI',
                              'fsim': 'FSIM',
                              'chi2': 'Chi2',
                              'mos': 'MOS'}, inplace=True)
      df_right.append(df_right_i)

    # Export to TeX code
    tex_code_left = []
    for df_i in df_left:
      tex_code_left.append(df_i.to_latex(index=False))
    tex_code_right = []
    for df_i in df_right:
      tex_code_right.append(df_i.to_latex(index=False))

    # Save to a .tex file
    with open(os.path.join(csv_file_folder, export_file_name), 'w') as f:
        for i, tex_i in enumerate(tex_code_left):
          f.write(tex_code_left[i])
          f.write("\n")
          f.write(tex_code_right[i])
          f.write("\n\n\n")


  # endregion


# region external functions

async def measure(mode='time', parms=None):
  """Asynchronous function which applies the measurement using two types of orchestration. 
  
  Args:
    mode (str): 
      Type of orchestration (default: 'time'):
      'time': Time-based orchestration. Given trigger times related to the measurement system, the function executes them with regard to the global trigger time, which starts when an external trigger was detected (e.g. on
      rising edge).
      'steps': Step-based orchestration. After specific counts of steps, the function executes the corresponding subfunctions for the measurement.
    parms (dict):
      Parameter file as a list, where the parameters are either the trigger time or the trigger steps for the measurement systems in the following order (default: None): [THz, FMCW, MV]

  Returns:

  Raises:
  """
  
  # Set up event loop (required for asynchronous function calls)
  loop = asyncio.get_running_loop()

  # Initialize variables
  ifi_1 = IFI()
  #thz_1 = THz(parms['thz']['init'])
  mv_1 = MV()
  results = [None] * 3

  # CONFIGURATION
  #==============

  # Configure UDP-Socket for communication with Raspberry Pie-based HW trigger
  sock = ifi_1.configure_udp_socket(parms)
  
  # Start matlab engine and make sure mmWaveStudio is opened. Then, run configuration of the radar, which has to be done only once at startup. This step is required to be done prior to the FMCW measurement.
  if parms['fmcw']['active']:
    me = ifi_1.matlab_mmwstudio_startup(parms['path_mmwstudio'])
  else:
    me = None

  # Start acquisition of THz system. This step is required to be done prior to the MV measurement.
  if parms['thz']['active']:
    # source, convert = thz_1.terasense_configuration(mode=0, 
    #                                                 config_name=parms['thz']['config_name'])
    # source = thz_1.terasense_start(source)
    #thz_1 = THz(num_shots=5)
    thz_1 = THz(parms['thz'])
    thz_1.start_stream()
  else:
    source, convert = None, None

  # Configure and connect to camera
  if parms['mv']['active']:
    camera, devices = mv_1.basler_camera_configuration(parms['mv'])
    camera = mv_1.basler_camera_connect(camera)
  else:
    camera = None

  # MEASUREMENTS
  #==============

  # Run measurement devices when corresponding trigger times are reached.
  print("----------------------------------------------------")
  print("Process running in infinite loop. Press 's' to stop.")
  inf_loop = True
  while inf_loop:

    print("Waiting for external trigger:")

    # Listen to UDP and run THZ, FMCW and MV measurements in parallel using own threads. The block trigger is set using a class attribute.
    tasks = [loop.run_in_executor(None, 
                                  ifi_1.listen_for_udp_trigger, 
                                  sock, parms),
            # loop.run_in_executor(None, 
            #                       ifi_1.thz_measurement, 
            #                       thz_1, source, convert, parms['thz']),
            loop.run_in_executor(None, 
                                  ifi_1.thz_measurement, 
                                  thz_1, parms['thz']),
            loop.run_in_executor(None, 
                                  ifi_1.fmcw_measurement,
                                  me,
                                  parms['fmcw']),
            loop.run_in_executor(None, 
                                  ifi_1.mv_measurement,
                                  mv_1, camera, devices, parms['mv'])]

    # Wait for results being returned by the tasks and extract
    print("All triggers raised. Waiting for results:")
    results = await asyncio.gather(*tasks)
    # img_thz = results[1][0]
    # data_thz = results[1][1]
    # source = results[1][2]
    # convert = results[1][3]
    # me = results[2][1]
    # img_mv = results[3][0]
    # camera = results[3][1]
    img_thz = results[1]
    data_thz = None
    me = results[2]
    img_mv = results[3][0]
    camera = results[3][1]

    # Save results
    print("Results received. Saving...")
    ifi_1.save_results([img_thz, data_thz, img_mv])
    print("Results saved.")

    # Ask user if new cycle desired
    ans = input("New cycle? y/n")
    if ans=='y':
      inf_loop = True
    else:
      inf_loop = False
      print('IFI application closed.')

  # Shut down image acquisition processes and/or disconnect
  source = thz_1.terasense_stop(source)
  camera = mv_1.basler_camera_disconnect(camera)

# endregion

#===============================================================================
if __name__ == '__main__':

  print(f'===')
  print(f'IFI')
  print(f'===')

  # region Parameters
  RUN_MEASURE = False
  RUN_ANALYSIS = True
  RUN_DETECTION = False
  MAKE_IMAGE_SIZE_EQUAL = False

  parms_measure = {'path_mmwstudio': "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\RunTime\\mmWaveStudio.exe",
            'trigger_steps': [],
            'udp_ip': "192.168.33.30",
            'udp_port': 12345,
            'udp_max_n_byte': 1024,
            'thz':{'active': True,
                   'trigger_time': 3.0,
                    'n_images': 10,
                    'mode': 0,
                    'config_name': 'Configs\\config_2025-02-04',
                    'save': False,
                    't_sleep': 0.1,
                    'init': {'LOAD_CONFIG': False,
                              'SAVE_CONFIG': False,
                              'LIVESTREAM_VISIBLE': False,
                              'MODE_BACKGROUND': 'RECORD',
                              'MODE_NORM': 'RECORD',
                              'MODE_DIFFERENCE': True,
                              'SET_SYNC_OUT': True,
                              'SET_EXTERNAL_SYNC': False,
                              'EXTERNAL_SYNC_EDGE': None,
                              'SYNC_DIVIDER': 1,
                              'SYNC_HOLDOFF': 0,
                              'LOAD_CONFIG_PATH': None,
                              'SAVE_CONFIG_PATH': None,
                              'FRAME_LENGTH': 256,
                              'ASPECT_RATIO': 2,
                              'RATE': 500,
                              'BELT_SPEED': 500.0,
                              'BACKGROUND_IDX': None,
                              'BACKGROUND_DATA': None,
                              'BACKGROUND_RECORD_COUNTS': 30,
                              'NORM_IDX': None,
                              'NORM_DATA_MASK': (None, None),
                              'THRESHOLD': 10.0,
                              'ACCUMULATION_ON': True,
                              'ACCUMULATION_LENGTH': 50,
                              'WHITE': 1.0,
                              'BLACK': 0.7,
                              'GAMMA': 0.2,
                              'SMOOTHNESS': 1}},
            'fmcw':{'active': False,
                    'trigger_time': 1.0},
            'mv': {'active': True,
                   'trigger_time': 18.5,
                   'mode': 'foreground_loop',
                   'n_images': 1,
                   'timeout_ms': 5000,
                   'show_stats': True,
                   'idx_device': 0,
                   'show_one_image': False,
                   'idx_show_one_image': 0}}

  # endregion

  parms_analysis_similarity = {'mode': 'common_similarity_metrics',
                    'folder_source': 'measurements\\measurements_thz_2025-10-20_2',
                    'folder_results': 'processed\\processed_similarity_thz',
                    'name_file_reference_sample': 'ifi_thz_reference.png',
                    'name_file_contaminated_sample': 'ifi_thz_contaminated.png',
                    'factors': {'cookie': ['chocolatewafer', 
                                'jellycookie', 
                                'waferroll', 
                                'vanillacrescent'], 
                                'dirt': ['glas', 
                                          'metal', 
                                          'pe', 
                                          'pp', 
                                          'pvc', 
                                          'ps'], 
                                'dirt_size': ['5', 
                                              '7'], 
                                'paper': ['0', 
                                          '10', 
                                          '20']},
                    'names_factors': {'cookie': 'Sample type',
                                      'dirt': 'LDFO material',
                                      'dirt_size': 'LDFO size',
                                      'paper':'Attenuation',
                                      'chocolatewafer': 'Chocolate Wafer',
                                      'jellycookie': 'Jelly Cookie',
                                      'vanillacrescent': 'Vanilla Crescent',
                                      'waferroll': 'Wafer Roll',
                                      'glas': 'Glass',
                                      'metal': 'Metal',
                                      'pe': 'PE',
                                      'pp': 'PP',
                                      'pvc': 'PVC',
                                      'ps': 'PS',
                                      '5': '5 mm',
                                      '7': '7 mm',
                                      '0': '0p',
                                      '10': '10p',
                                      '20': '20p'},
                      'variables': ['ssim', 
                                    'cwssim',
                                    'mse', 
                                    'psnr',
                                    'are', 
                                    'voi',
                                    'fsim',
                                    'chi2',
                                    'mos'],
                      'names_variables': 
                      ['SSIM', # Structural Similarity Index
                       'CWSSIM', # Complex Wavelet Structural Similarity Index
                       'MSE', # Mean Squared Error
                       'PSNR', # Peak-Signal-to-Noise Ratio
                       'ARE', # Adapted Rand Error
                       'VOI', # Variation of Information
                       'FSIM', # FSIM
                       'CHI2', # Chi-Squared
                       'MOS'], # Variation of Information
                      'print_head_tail': False,
                      'delta_list': [0.1, 0.1, 0.1, 0.1]}

  # endregion

  # region Application

  # Run asynchronous measurements in one function
  if RUN_MEASURE:
    asyncio.run(measure(mode='time', parms=parms_measure))

  # Create global object
  ifi_2 = IFI()

  # Run analysis of acquired images
  if RUN_ANALYSIS:

    # Make image size equal if required
    if MAKE_IMAGE_SIZE_EQUAL:
      ifi_2.make_image_size_equal('measurements\\measurements_thz_2025-10-20', 
                                  'measurements\\measurements_thz_2025-10-20_2', parms_analysis_similarity['name_file_reference_sample'], parms_analysis_similarity['name_file_contaminated_sample'])

    # Analyze similarity of reference and sample images
    folder_results_date = ifi_2.analyze_image_similarity(parms_analysis_similarity['folder_source'], 
      parms_analysis_similarity['folder_results'],
      parms_analysis_similarity['name_file_reference_sample'], parms_analysis_similarity['name_file_contaminated_sample'], 
      parms_analysis_similarity['mode'],
      parms_analysis_similarity['delta_list'])
    
    # Convert results to TeX-table
    ifi_2.convert_csv_files_to_tex_table(folder_results_date, 
                                         'table_results.tex')
    
    # Calculate statistics of similarity indices
    ifi_2.calculate_statistics_of_similarity(csv_file_path=folder_results_date, 
                                             parms=parms_analysis_similarity, save_stats_to_tex=True)
    
    # Calculate statistical analysis based on ANOVA
    ifi_2.calculate_anova(csv_file_path=folder_results_date)


  # Run classification or anomaly detection
  if RUN_DETECTION:
    raise NotImplementedError("This feature is not yet implemented.")
    # ifi_2.classify_detect(path_dataset='measurements\\measurements_thz_2025-10-20_2')
  
  # endregion