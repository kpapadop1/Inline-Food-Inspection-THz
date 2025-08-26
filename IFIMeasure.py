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

DATE
  2024-December-19

LAST MODIFIED
  -

V1.0 / Copyright 2022 - Konstantinos Papadopoulos
-------------------------------------------------------------------------------
Notes
------
Run the requirements file using the command "pip install -r requirements.txt"

or the commands below (after adapting the paths):

pip install asyncio
pip install pillow
pip install wheel
pip install "C:\\WORK\CAISA\\Promotion\\4 - Entwicklung\\Hardware\\Radarsensorik\\TeraSense\\TeraFAST-4.3.0-full - NEU\\packages\\numpy-1.24.1-cp310-cp310-win_amd64.whl"
pip install "C:\\WORK\CAISA\\Promotion\\4 - Entwicklung\\Hardware\\Radarsensorik\\TeraSense\\TeraFAST-4.3.0-full - NEU\\packages\\opencv_python-4.7.0.68-cp37-abi3-win_amd64.whl"
pip install "C:\\WORK\CAISA\\Promotion\\4 - Entwicklung\\Hardware\\Radarsensorik\\TeraSense\\TeraFAST-4.3.0-full - NEU\\packages\\Pillow-9.4.0-cp310-cp310-win_amd64.whl"
pip install "C:\\WORK\CAISA\\Promotion\\4 - Entwicklung\\Hardware\\Radarsensorik\\TeraSense\\TeraFAST-4.3.0-full - NEU\\packages\\pyserial-3.5-py2.py3-none-any.whl"
pip install "C:\\WORK\CAISA\\Promotion\\4 - Entwicklung\\Hardware\\Radarsensorik\\TeraSense\\TeraFAST-4.3.0-full - NEU\\packages\\six-1.16.0-py2.py3-none-any.whl"
pip install "C:\\WORK\CAISA\\Promotion\\4 - Entwicklung\\Hardware\\Radarsensorik\\TeraSense\\TeraFAST-4.3.0-full - NEU\\packages\\wxPython-4.2.0-cp310-cp310-win_amd64.whl"
pip install "C:\\WORK\CAISA\\Promotion\\4 - Entwicklung\\Hardware\\Radarsensorik\\TeraSense\\TeraFAST-4.2.5-py3-none-win_amd64.whl"
pip install pypylon
pip install keyboard
pip install pynput
pip install opencv-python
pip install matplotlib
pip install matlabengine==9.14.2
pip install scikit-learn
pip install ultralytics
pip install datasets
pip install tensorflow=2.14.0

Todo:
------
- Debug all functions!
- THz: Add more parameters. Add 'parms' to pass them all at once!
- See backup of 24.06.2025
  
-------------------------------------------------------------------------------
"""
# region constants

# TODO: Adapt paths if required!
path_thz = 'c:\\WORK\\CAISA\\Promotion\\4 - Entwicklung\\Software\\Python\\Terahertz Sensing'     # Path to THz class file
path_fmcw = r'C:\Users\julia\Documents\RADAR\FMCW'        # Path to FMCW class file
path_mv = r'C:\Users\julia\Documents\Thz\Mv_git\Machine-Vision'         # Path to MV class file

# endregion


# region dependencies

# Miscellaneous
import asyncio
import time
import matplotlib.pyplot as plt
import socket
import os
import sys
import subprocess
import json

# THz-based
import numpy as np
from PIL import Image, ImageEnhance
import sys, os
import pickle
from datetime import datetime
import keyboard
import matlab.engine
sys.path.append(path_thz)
#from THz import THz
from THz_2 import THz

# FMCW-based
sys.path.append(path_fmcw)
from Fmcw import Fmcw

# MV-based
from pypylon import pylon
import cv2
sys.path.append(path_mv)
from MV import MV
from skimage.filters import threshold_otsu

# ML /DL
# import tensorflow as tf
# from tensorflow import keras
# from keras import ops
# from keras.layers import (Input, Conv2D, Flatten, Dense, 
#                           Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, UpSampling2D, Concatenate, MultiHeadAttention, Add, Subtract, Multiply)
# from keras.models import Model, Sequential
# from keras.optimizers import SGD, Adam
# from keras.regularizers import l2
# from keras.preprocessing.image import ImageDataGenerator
# from keras.losses import CategoricalCrossentropy
# from keras import backend as K
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import roc_curve, auc

# endregion


#==============================================================================
#%% CLASS DEFINITION
#==============================================================================

class IFIMeasure:
  """This class implements all classes and methods for the execution of specific measurements for Inline Food Inspection.
  
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
          print('THz measurement starting....')
          # (img_thz, 
          #  data_thz, 
          #  source, 
          #  convert) = thz_obj.execute_measurement_simple(source, 
          #                                         convert,
          #                                         n_images=parms_thz['n_images'], 
          #                                         save=parms_thz['save'],
          #                                         t_sleep=parms_thz['t_sleep'])
          raw_thz = thz_obj.execute_measurement_simple()
        else:
          print('THz measurement deactivated.')
          img_thz = 0

        print(f"THz duration: {time.time() - t_0} s.")
        break

    return raw_thz #, data_thz, source, convert

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

          with open(os.path.join(folder_path, filename_prefix + self.MEASUREMENT_TECHNOLOGIES[i] + '_' + formatted_time + '.npy'), 'wb') as file:
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

  # endregion


# region External functions

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
  ifi_1 = IFIMeasure()
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
    #img_thz = results[1][0]
    data_thz = results[1][0]    #raw data from thz
    data_thz_raw = results[1][1]  #raw data from thz
    me = results[2]
    img_mv = results[3][0]
    camera = results[3][1]

    # Save results
    print("Results received. Saving...")
    ifi_1.save_results([data_thz, data_thz_raw , img_mv])
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

  parms_measure = {'path_mmwstudio': "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\RunTime\\mmWaveStudio.exe",
            'trigger_steps': [],
            'udp_ip': "192.168.33.30",
            'udp_port': 12345,
            'udp_max_n_byte': 1024,
            'thz':{'active': True,
                   'trigger_time': 3.5,
                    'n_images': 1,
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
                              'FRAME_LENGTH': 2048,
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
                   'timeout_ms': 2000,
                   'show_stats': True,
                   'idx_device': 0,
                   'show_one_image': False,
                   'idx_show_one_image': 0}}

  parms_analysis = {}

  # endregion

  # region Application

  # Run asynchronous measurements in one function
  asyncio.run(measure(mode='time', parms=parms_measure))

  # endregion

  pass

