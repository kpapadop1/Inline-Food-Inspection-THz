# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script defines the class for the application of Computer Vision methods.
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script defines the class for the application of Computer Vision methods.


SEE ALSO
  -
  
FILE
  .../CvUnit.py

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
[INFO: WORKS WITH opencv-python == 4.9.0.80]
[INSTALL WITH opencv-contrib-python== 4.10.0.84]

After installing Python v3 and PIP > 19.0 apply following installation commands
pip install numpy
pip install opencv-python
pip install glob2
pip install matplotlib
pip install pillow
pip install ultralytics
pip install datasets
pip install --upgrade tensorflow
pip install scikit-learn 
pip install scikit-image

Todo:
------
- Implement methods from computational photography and ML
  
-------------------------------------------------------------------------------
"""
#==============================================================================
#%% DEPENDENCIES
#==============================================================================
import os
import time
from datetime import datetime
import math
import numpy as np
import cv2 as cv
import glob
import random
from matplotlib import pyplot as plt
from matplotlib import colors as clrs
from PIL import Image
import sklearn.model_selection
from ultralytics import YOLO
from datasets import load_dataset
import tensorflow as tf
import keras
import sklearn
from pypylon import pylon
from auxiliary.preprocess import check_images, convert_batchdataset_to_numpy
from auxiliary.postprocess import plot_history, plot_confusion_mat
from auxiliary.build_cnn_model import build_cnn_model
from auxiliary.build_rnn_model import build_rnn_model
from auxiliary.build_ae_model import build_ae_model
from auxiliary.build_cgan_model import (build_cgan_model, train_cgan_model)

#==============================================================================
#%% CLASS DEFINITION
#==============================================================================

class MV:
    #-------------
    # Parameters
    #-------------
    
    #-------------
    # Methods
    #-------------
    def __init__(self):
        pass

    # Basic operations
    #-----------------
    def load_image(self, img_str):
      '''Load image and trigger error if file does not exist.
      
      Inputs
      ------
      img_str:  Local path to string / String

      Outputs
      -------
      img:      Read image / image object
      '''
        
      img = cv.imread(img_str)
      assert img is not None, "file could not be read, check with os.path.exists()"
      return img
    

    def show_image(self, img, window_name='Image', type='cv'):
      '''Show image using CV's method (b, g, r) or Matplotlib's method (r, g, b).
      
      Inputs
      ------
      img:          Local path to image / String
      window_name:  Window name / String
      type:         Typ ('cv': CV; else: Matplotlib)

      Outputs:
      --------
      '''

      if type == 'cv':
        cv.imshow(window_name, img)
      else:
         plt.imshow(img)


    def read_pixel_slow(self, img_str, x, y):
      '''Read pixel (color) values using a slow access method.

      Inputs
      ------
      img_str:          Local path to image / String
      x:                Pixel abscissa (x) / 1
      y:                Pixel ordinate (y) / 1

      Outputs:
      --------
      pixel_val:        Pixel content (color values) / 1
      '''

      img = mv_1.load_image(img_str)
      pixel_val = img[x, y]
      return pixel_val
    

    def write_pixel_slow(self, img_str, x, y, cols):
      '''Write color values to pixel using a slow access method.

      Inputs
      ------
      img_str:          Local path to image / String
      x:                Pixel abscissa (x) / 1
      y:                Pixel ordinate (y) / 1
      pixel_val:        Pixel content (color values) / 1             

      Outputs:
      --------
      
      '''

      img = mv_1.load_image(img_str)
      img[x, y] = cols 
      return img


    def read_pixel_fast(self, img_str, x, y):
      '''Read pixel (color) values using an optimized access method.

      Inputs
      ------
      img_str:          Local path to image / String
      x:                Pixel abscissa (x) / 1
      y:                Pixel ordinate (y) / 1

      Outputs:
      --------
      pixel_val         Pixel values (color values) / 1
      '''

      img = mv_1.load_image(img_str)
      r = img.item(x,y,0)
      g = img.item(x,y,1)
      b = img.item(x,y,2)
      pixel_val = [r, g, b]
      return pixel_val
    

    def write_pixel_fast(self, img_str, x, y, cols):
      '''Write pixel (color) values using an optimized access method.

      Inputs
      ------
      img_str:          Local path to image / String
      x:                Pixel abscissa (x) / 1
      y:                Pixel ordinate (y) / 1
      pixel_val:        Pixel content (color values) / 1

      Outputs:
      --------
      '''

      img = mv_1.load_image(img_str)
      img.itemset((x,y,0),cols[0])
      img.itemset((x,y,1),cols[1])
      img.itemset((x,y,2),cols[2])

      return img
    

    def get_image_properties(self, img_str):
      '''Get basic image properties like shape, size, and data type.

      Inputs
      ------
      img_str:          Local path to image / String

      Outputs:
      --------
      shape:            Shape of image / 1
      size:             Size of image / 1
      dtype:            Type of image / [type]
      '''

      img = mv_1.load_image(img_str)
      shape = img.shape
      size = img.size
      dtype = img.dtype

      return shape, size, dtype
    

    def extract_roi(self, img_str, x_rng, y_rng):
      '''Get region of interest (ROI).

      Inputs
      ------
      img_str:          Local path to image / String
      x_rng:            Range in x-direction for extraction as 2-element list / 1 
      y_rng:            Range in y-direction for extraction as 2-element list / 1 

      Outputs:
      --------
      roi:              Image extract (range-of-interest) / image
      '''

      img = mv_1.load_image(img_str)
      roi = img[y_rng[0]:y_rng[1], x_rng[0]:x_rng[1]]
      return roi


    def place_roi(self, img_str, roi, x_rng, y_rng):
      '''Place region of interest (ROI) into a specific image.

      Inputs
      ------
      img_str:          Local path to image / String
      roi:              Image extract to be placed into image (range-of-interest) / image
      x_rng:            Range in x-direction for extraction as 2-element list / 1
      y_rng:            Range in y-direction for extraction as 2-element list / 1 

      Outputs:
      --------
      img:              Image with image extract placed into
      '''

      img = mv_1.load_image(img_str)
      img[y_rng[0]:y_rng[1], x_rng[0]:x_rng[1]] = roi
      return img
    

    def split_channels(self, img_str):
       '''Split color channels (b, g, r).

      Inputs
      ------
      img_str:          Local path to image / String

      Outputs:
      --------
      channels:         Color channels / Arrays
       '''

       img = mv_1.load_image(img_str)
       b, g, r = cv.split(img)
       channels = [b, g, r]

       return channels


    def merge_channels(self, channels):
       '''Merge color channels (b, g, r).

      Inputs
      ------
      channels:          Color channels to merge / Channels

      Outputs:
      --------
      img:              Image with merged channels / Image
       '''

       img = cv.merge((channels[0], channels[1], channels[2]))
       return img
    

    def make_border(self, img_str, border_widths, type=cv.BORDER_CONSTANT, values=[255, 255, 255]):
      '''Make border around image of a specific type.

      Inputs
      ------
      img_str:          Local path to image / String
      border_widths:    Border width / 1
      type:             Type of border / Parameter
      values:           Color / 1

      Outputs:
      --------
      image_framed:     Framed image / Image
      '''

      img = mv_1.load_image(img_str)
      image_framed = cv.copyMakeBorder(img, border_widths[0], 
                                      border_widths[1], 
                                      border_widths[2], 
                                      border_widths[3],
                                      cv.BORDER_CONSTANT,
                                      value=values)
      return image_framed


    # Arithmetic operations
    #----------------------
    def add_images(self, img_str_1, img_str_2):
      '''Add images by adding the data channel-wise. Note, that this is a saturated operation.

      Inputs
      ------
      img_str_1:          Local path to first image to add / String
      img_str_2:          Local path to second image to add / String

      Outputs:
      --------
      img_add:            Added image / Image
      '''

      img_1 = mv_1.load_image(img_str_1)
      img_2 = mv_1.load_image(img_str_2)
      img_add = cv.add(img_1, img_2)
      return img_add


    def blend_images(self, img_str_1, img_str_2, alpha, beta, gamma):
      '''Blend images based on given weights based on the formula: dst = alpha_1*img_1 + beta*img_2 + gamma.

      Inputs
      ------
      img_str_1:          Local path to first image to blend / String
      img_str_2:          Local path to second image to blend / String
      alpha:              Alpha-value for blending / 1.0
      beta:               Beta-value for blending / 1.0

      Outputs:
      --------
      '''

      img_1 = mv_1.load_image(img_str_1)
      img_2 = mv_1.load_image(img_str_2)
      img_blend = cv.addWeighted(img_1, alpha, img_2, beta, gamma)
      return img_blend
    

    def overlay_images_using_mask(self, img_str_1, img_str_2):
      '''Overlay images using mask from second image after converting it to B&W based on bitwise operations.

      Inputs
      ------
      img_str_1:          Local path to first image to overlay / String
      img_str_2:          Local path to second image to overlay / String

      Outputs:
      --------
      '''

      # Put mask image on top-left corner. For this, start by creating a ROI of the background image using 
      # the size of the overlay image
      img_1 = mv_1.load_image(img_str_1)
      img_2 = mv_1.load_image(img_str_2)
      rows, cols, channels = img_2.shape
      roi = img_1[0:rows, 0:cols]
      
      # Create a mask out of the overlay image and create its inverse mask also
      img_2_gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
      ret, mask = cv.threshold(img_2_gray, 10, 255, cv.THRESH_BINARY)
      mask_inv = cv.bitwise_not(mask)
      
      # Black-out the area of the overlay image in ROI of the backgound image
      img1_bg = cv.bitwise_and(roi, roi, mask = mask_inv)
      
      # Take only region of logo from overlay image
      img2_fg = cv.bitwise_and(img_2, img_2, mask = mask)
      
      # Put overlay image in ROI and modify the main image
      dst = cv.add(img1_bg, img2_fg)
      img_1[0:rows, 0:cols ] = dst
      return img_1


    # Image Processing
    #-----------------
    def extract_image_using_color_range(self, img_str, hsv_ub, hsv_lb):
      '''Extract image sections of (b,g,r)-image based on HSV color range.

      Inputs
      ------
      img_str:          Local path to image / String
      hsv_ub:           Upper bound of HSV values / 1
      hsv_lb:           Lower bound of HSV values / 1

      Outputs:
      --------
      img_extract       Image extract / Image
      '''

      # Convert BGR to HSV
      img = mv_1.load_image(img_str)
      hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
      
      # Define range of blue color in HSV
      upper_blue = np.array(hsv_ub)
      lower_blue = np.array(hsv_lb)
      
      # Threshold the HSV image to get only blue colors
      mask = cv.inRange(hsv, lower_blue, upper_blue)
      
      # Bitwise-AND mask and original image
      img_extract = cv.bitwise_and(img, img, mask=mask)
      return img_extract


    def scale(self, img_str, coeff_scale_x=2, coeff_scale_y=2, show_results=True):
      '''Scale input image. INTER_CUBIC mode is selected as default (other: INTER_LINEAR, INTER_AREA).
      
      Inputs
      ------
      img_str:           Local path to image as string / String
      coeff_scale_x:     Coefficient for scaling in x-direction (default: 2) / 1
      coeff_scale_y:     Coefficient for scaling in x-direction (default: 2 / 1
      show_results:      Flag for plotting results (default: True) / True; False

      Outputs
      ------
      img_scaled:        Processed image / Image
      '''

      img_original = cv.imread(img_str)
      img_scaled = cv.resize(img_original, None, fx=coeff_scale_x, fy=coeff_scale_y, interpolation=cv.INTER_CUBIC)

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(img_scaled)
        plt.title('Scaled image')
        plt.show()

      return img_scaled


    def translate(self, img_str, delta_x=100, delta_y=50, show_results=True):
      '''Translate input image.
      
      Inputs
      ------
      img_str:           Local path to image as string / String
      delta_x:           Displacement in x-direction (default: 100) / 1
      delta_y:           Displacement in x-direction (default: 50) / 1
      show_results:      Flag for plotting results (default: True) / True; False

      Outputs
      ------
      img_translated:    Processed image / Image
      '''
      
      img_original = cv.imread(img_str)
      rows, cols, _ = img_original.shape
      M = np.float32([[1, 0, delta_x], 
                      [0, 1, delta_y]])
      img_translated = cv.warpAffine(img_original, M, (cols, rows))

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(img_translated)
        plt.title('Translated image')
        plt.show()

      return img_translated


    def rotate(self, img_str, center=None, angle=90, scale=1, show_results=True):
      '''Rotate image.
      
      Inputs
      ------
      img_str:           Local path to image as string
      center:            Center point (default: middle point) / 1
      angle:             Angle (default: 90°) / °
      scale:             Scale coefficient (default: 1) / 1
      show_results:      Flag for plotting results (default: True) / True; False

      Outputs
      ------
      img_rotated:       Processed image / Image
      '''

      img_original = cv.imread(img_str)
      rows, cols, _ = img_original.shape
    
      if center is None:
        center = ((cols - 1)/2.0, (rows - 1)/2.0)
    
      M = cv.getRotationMatrix2D(center, angle, scale)
      img_rotated = cv.warpAffine(img_original, M, (cols, rows))

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(img_rotated)
        plt.title('Translated image')
        plt.show()

      return img_rotated


    def transform_affine(self, img_str, pts_in=None, pts_out=None, show_results=True):
      '''Apply affine transformation to image. In an affine transformation, parallel line remain parallel.
      
      Inputs
      ------
      img_str:           Local path to image as string / String
      pts_in:            Input point (x,y) set consisting of three elements (default: np.float32([[50, 50], [200, 50], [50, 200]])) / 1
      pts_out:           Output point (x,y) set consisting of three elements (default: np.float32([[10, 100], [200, 50], [100, 250]])) / 1
      show_results:      Flag for plotting results (default: True) / True; False

      Outputs
      ------
      transform_affine:  Processed image / Image
      '''

      # Apply defaults if required
      if pts_in is None:
        pts_in = np.float32([[50, 50], [200, 50], [50, 200]])
      if pts_out is None:
        pts_out = np.float32([[10, 100], [200, 50], [100, 250]])

      # Apply operations (load, get matrix of transformation and apply)
      img_original = cv.imread(img_str)
      rows, cols, _ = img_original.shape
      M = cv.getAffineTransform(pts_in, pts_out)
      img_transformed_affine = cv.warpAffine(img_original, M,(cols, rows))

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(img_transformed_affine)
        plt.title('Transformed (affine) image')
        plt.show()

      return img_transformed_affine


    def transform_perspective(self, img_str, pts_in=None, pts_out=None, size=None, show_results=True):
      '''Apply perspective transformation.

      Inputs
      ------
      img_str:                      Local path to image as string / String
      pts_in:                       Input point (x,y) set consisting of four elements (default: np.float32([[56, 65], [368, 52], [28, 387],[389, 390]])) / 1
      pts_out:                      Output point (x,y) set consisting of four elements (default: np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])) / 1
      size:                         Tupel of size (x,y) (default: (300, 300)) / 1
      show_results:                 Flag for plotting results (default: True) / True; False

      Outputs
      ------
      img_transformed_perspective:  Processed image / Image
      '''

      # Apply defaults if required
      if pts_in is None:
        pts_in = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
      if pts_out is None:
        pts_out = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
      if size is None:
        size = (300, 300)

      # Apply operations (load, get matrix of transformation and apply)
      img_original = cv.imread(img_str)
      rows, cols, _ = img_original.shape
      M = cv.getPerspectiveTransform(pts_in, pts_out)
      img_transformed_perspective = cv.warpPerspective(img_original, M, size)

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(img_transformed_perspective)
        plt.title('Transformed (perspective) image')
        plt.show()
      
      return img_transformed_perspective


    def convert_color_space(self, img_str, type):
      '''Convert color space.
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      type:                           Type of conversion (e.g. 'RGB 2 BGR', 
                                                               'BGR 2 RGB', 
                                                               'RGB 2 GRAY', 
                                                               'BGR 2 GRAY', 
                                                               'RGB 2 HSV', 
                                                               'BGR 2 HSV', 
                                                               'HSV 2 RGB',
                                                               'HSV 2 BGR' etc.) / - 

      Outputs
      ------
      img_col_space_converted         Image after color space conversion / Image
      
      '''
      img_original = cv.imread(img_str)
      if type=='':
        img_color_space_converted = cv.cvtColor(img_original, cv.COLOR_RGB2BGR)
      elif type=='':
        img_color_space_converted = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
      elif type=='':
        img_color_space_converted = cv.cvtColor(img_original, cv.COLOR_RGB2GRAY)
      elif type=='':
        img_color_space_converted = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
      elif type=='':
        img_color_space_converted = cv.cvtColor(img_original, cv.COLOR_RGB2HSV)
      elif type=='':
        img_color_space_converted = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)
      elif type=='':
        img_color_space_converted = cv.cvtColor(img_original, cv.COLOR_HSV2RGB)
      elif type=='':
        img_color_space_converted = cv.cvtColor(img_original, cv.COLOR_HSV2BGR)
      else:
        img_color_space_converted = cv.cvtColor(img_original, type)

      return img_color_space_converted


    def threshold_simple(self, img_str, threshold_val=127, substitute_val=255, type='BINARY', show_results=True):
      '''Simple thresholding using one global threshold value. Every pixel that exceeds this value is replaced by a specific value
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      threshold_val:                  Threshold value (default: 127) / 1
      substitute_val:                 Substitute value (default: 255) / 1
      type:                           Type of thresholding ('BINARY', 'BINARY INV', 'TRUNC', 'TOZERO', 'TOZERO INV'; default: 'BINARY')
      show_results:                   Show results (default: True) / True; False
      
      Outputs
      -------
      image_thresholded:              Processed image / Image
      '''

      img_original = cv.imread(img_str)

      if type == 'BINARY':
        ret, image_thresholded = cv.threshold(img_original, threshold_val, substitute_val, cv.THRESH_BINARY)
      elif type == 'BINARY INV':
        ret, image_thresholded = cv.threshold(img_original, threshold_val, substitute_val, cv.THRESH_BINARY_INV)
      elif type == 'TRUNC':
        ret, image_thresholded = cv.threshold(img_original, threshold_val, substitute_val, cv.THRESH_TRUNC)
      elif type == 'TOZERO':
        ret, image_thresholded = cv.threshold(img_original, threshold_val, substitute_val, cv.THRESH_TOZERO)
      elif type == 'TOZERO INV':
        ret, image_thresholded = cv.threshold(img_original, threshold_val, substitute_val, cv.THRESH_TOZERO_INV)

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(image_thresholded)
        plt.title('Thresholded image')
        plt.show()

      return image_thresholded
    

    def threshold_adaptive(self, img_str, threshold_val=255, adaptive_method='ADAPTIVE THRESH MEAN', type='BINARY', size_neigborhood=11, const_subtracted=2, show_results=True):
      '''Apply adaptive threshold. This technique takes into regard the neigborhood of the pixel in question.
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      threshold_val:                  Threshold value (default: 255) / 1.0
      adaptive_method:                Adaptive threshold method ('ADAPTIVE THRESH MEAN', 'ADAPTIVE THRESH GAUSSIAN C'; default: ADAPTIVE THRESH MEAN) / String
      type:                           Type of thresholding ('BINARY', 'BINARY INV'; default: 'BINARY') / String
      size_neigborhood:               Size of pixel neigborhood (odd number; default: 11) / 1
      const_subtracted:               Constant, which is subtracted from the mean (usually a positive value; default: 2) / 1
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      image_thresholded:              Processed image / Image
      '''

      img_original = cv.imread(img_str, cv.IMREAD_GRAYSCALE)
      img_original = img_original.astype(np.uint8)

      if adaptive_method == 'ADAPTIVE THRESH MEAN':
        _adaptive_method = cv.ADAPTIVE_THRESH_MEAN_C
      elif type == 'ADAPTIVE THRESH GAUSSIAN C':
        _adaptive_method = cv.ADAPTIVE_THRESH_GAUSSIAN_C

      if type == 'BINARY':
        _type = cv.THRESH_BINARY
      elif type == 'BINARY INV':
        _type = cv.THRESH_BINARY_INV

      image_thresholded = cv.adaptiveThreshold(img_original, threshold_val, _adaptive_method, _type, size_neigborhood, const_subtracted)

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(image_thresholded)
        plt.title('Thresholded image')
        plt.show()

      return image_thresholded
    

    def threshold_otsu(self, img_str, threshold_val=127, substitute_val=255, show_results=True):
      '''Apply adaptive threshold based on Otsu's method. This method avoids havin to set an empirical 
      value for thresholding. Instead, it takes the value between classes that minimizes the variance within these classes.
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      threshold_val:                  Threshold value (default: 127) / 1
      substitute_val:                 Substitute value (default: 255) / 1
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      image_thresholded:              Processed image / Image
      '''

      img_original = cv.imread(img_str)

      ret, image_thresholded = cv.threshold(img_original, threshold_val, substitute_val, cv.THRESH_BINARY+cv.THRESH_OTSU)

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(image_thresholded)
        plt.title('Thresholded image')
        
        plt.show()

      return image_thresholded
    

    def image_filtering_2d_convolution(self, img_str, kernel=np.ones((5, 5), np.float32)/25, show_results=True):
      '''Perform image filtering by applying a 2D convolution for averaging.
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      kernel:                         Kernel (default: np.ones((5, 5), np.float32)/25) / 1
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      image_filtered:                 Processed image / Image
      '''

      img_original = cv.imread(img_str)

      image_filtered = cv.filter2D(img_original, -1, kernel)

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(image_filtered)
        plt.title('Filtered (2D conv., averaged) image')
        plt.show()

      return image_filtered
    

    def image_smoothing(self, img_str, mode='AVERAGE', parameter_list=[], show_results=True):
      '''Smooth images based on different modes.
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      mode:                           Mode and parameter list:
                                      'AVERAGE': parameter_list=[(5, 5)]
                                      'GAUSSIAN': parameter_list=[(5, 5), 0]
                                      'MEDIAN': parameter_list=[5]
                                      'BILATERAL': parameter_list=[9, 75, 75]
                                      (default: 'AVERAGE') / -
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      image_filtered:                 Processed image / Image
      '''

      img_original = cv.imread(img_str)

      if parameter_list is None:
        raise TypeError("Parameter list is missing.")

      if mode == 'AVERAGE':
        image_filtered = cv.blur(img_original, parameter_list)
      elif mode == 'GAUSSIAN':
        image_filtered = cv.GaussianBlur(img_original, parameter_list[0], parameter_list[1])
      elif mode == 'MEDIAN':
        image_filtered = cv.medianBlur(img_original, parameter_list[0])
      elif mode == 'BILATERAL':
        image_filtered = cv.bilateralFilter(img_original, parameter_list[0], parameter_list[1], parameter_list[2])
      else:
        raise TypeError("Mode unknown.")
        
      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(image_filtered)
        plt.title('Filtered (2D conv., averaged) image')
        plt.show()

      return image_filtered
    

    def morphological_operation(self, img_str, mode='EROSION', kernel=np.ones((5,5),np.uint8), kernel_type='PASSED', kernel_size=(5, 5), show_results=True):
      '''Apply morphological operation to image based on the mode and kernel.
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      mode:                           Mode as string (default: 'EROSION') / -:
                                      'EROSION'
                                      'DILATION'
                                      'OPENING'
                                      'CLOSING'
                                      'GRADIENT'
                                      'TOP HAT'
                                      'BLACK HAT'
      kernel:                         Kernel / Numpy array
      kernel_type:                    Type of kernel, which is constructed inside the function (default: 'PASSED'):
                                      'PASSED': As passed by the parameter 'kernel'
                                      'RECTANGULAR': Rectangular shape.
                                      'ELLIPSE': Elliptical shape.
                                      'CROSS': Cross-shaped kernel.
      kernel_size:                    Kernel size if other than 'PASSED' as tupe (default: (5, 5)) / 1
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      image_processed:                Processed image / Image
      '''

      img_original = cv.imread(img_str)

      if kernel_type == 'PASSED':
        _kernel = kernel
      elif kernel_type == 'RECTANGULAR':
        _kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
      elif kernel_type == 'ELLIPSE':
        _kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
      elif kernel_type == 'CROSS':
        _kernel = cv.getStructuringElement(cv.MORPH_CROSS, kernel_size)
      else:
        raise TypeError("Unkown kernel type.")

      if mode == 'EROSION':
        image_processed = cv.erode(img_original, _kernel, iterations=1)
      elif mode == 'DILATION':
        image_processed = cv.dilate(img_original, _kernel, iterations=1)
      elif mode == 'OPENING':
        image_processed = cv.morphologyEx(img_original, cv.MORPH_OPEN, _kernel)
      elif mode == 'CLOSING':
        image_processed = cv.morphologyEx(img_original, cv.MORPH_CLOSE, _kernel)
      elif mode == 'GRADIENT':
        image_processed = cv.morphologyEx(img_original, cv.MORPH_GRADIENT, _kernel)
      elif mode == 'TOP HAT':
        image_processed = cv.morphologyEx(img_original, cv.MORPH_TOPHAT, _kernel)
      elif mode == 'BLACK HAT':
        image_processed = cv.morphologyEx(img_original, cv.MORPH_BLACKHAT, _kernel)
      else:
        raise TypeError("Mode unknown.")

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(image_processed)
        plt.title('Processed image (mode: {mode})')
        plt.show()

      return image_processed


    def image_gradient(self, img_str, kernel_type='LAPLACIAN', kernel_size=5, show_results=True):
      '''Get image gradient of image based on the mode and kernel size.
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      kernel_type:                    Type of kernel, which is constructed inside the function (default: 'PASSED'):
                                      'LAPLACIAN': Kernel based on Laplacian derivatives
                                      'SOBEL X': Sobel kernel in x-direction
                                      'SOBEL Y': Sobel kernel in y-direction
                                      'SOBEL XY': Sobel kernel in x- and y-direction
      kernel_size:                    Kernel size (default: 5) / 1
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      image_processed:                Processed image / Image
      '''

      img_original = cv.imread(img_str)

      if kernel_type == 'LAPLACIAN':
        image_gradient = cv.Laplacian(img_original, cv.CV_64F)
      elif kernel_type == 'SOBEL X':
        image_gradient = cv.Sobel(img_original, cv.CV_64F, 1, 0, ksize=kernel_size)
      elif kernel_type == 'SOBEL Y':
        image_gradient = cv.Sobel(img_original, cv.CV_64F, 0, 1, ksize=kernel_size)
      elif kernel_type == 'SOBEL XY':
        image_gradient = cv.Sobel(img_original, cv.CV_64F, 1, 1, ksize=kernel_size)
      else:
        raise TypeError("Kernel type unknown.")
      
      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(image_gradient)
        plt.title('Processed image (mode: {mode})')
        plt.show()

      return image_gradient


    def image_pyramids(self, img_str, level=5, direction='DOWN', show_results=True):
      '''Get the image pyramids based on the given level.
      
      Inputs
      ------
      img_str:                        Local path to image as string / String
      level:                          Level depth as integer (default: 5) / 1
      direction:                      Direction (default: 'DOWN'):
                                      'DOWN': Get image(s) of lower resolution
                                      'UP': Get image(s) of higher resolution
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      image_processed_array:          List of processed images / Image
      '''

      img_original = cv.imread(img_str)

      image_processed_array = [img_original]
      for i in range(level):
        if direction=='DOWN':
          image_processed_array.append(cv.pyrDown(image_processed_array[-1]))
        elif direction=='UP':
          image_processed_array.append(cv.pyrUp(image_processed_array[-1]))
        else:
          raise TypeError("Kernel type unknown.")
        
      # Plot
      if show_results:

        if level <= 10:
          n_cols = int(math.ceil(math.sqrt(level)))
          n_rows = int(math.ceil(level / n_cols))
          fig = plt.figure(figsize=(10, 8))
          for i in range(level):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            ax.imshow(image_processed_array[i])
            ax.set_title(f'Image (level {i})')
        else:
          print('image_pyramids(): Only up to 10 images can be displayed using this function.')
        
        plt.show()

      return image_processed_array


    def image_blend(self, img_str_1, img_str_2, alpha=0.5, beta=0.5, gamma=0, show_results=True):
      '''Blend two images together using weightings.
            
      Inputs
      ------
      img_str_1:                      Local path to first image as string / String
      img_str_2:                      Local path to second image as string / String
      alpha:                          Weight of the first image (0...1) (default: 0.5) / 1.0
      beta:                           Weight of the second image (0...1) (default: 0.5) / 1.0
      gamma:                          Scalar added to the sum (default: 0) / 1.0
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      image_blended:                  Blended image / Image
      '''

      img_original_1 = cv.imread(img_str_1)
      img_original_2 = cv.imread(img_str_2)

      # Adapt sizes
      width1, height1, _ = img_original_1.shape
      width2, height2, _ = img_original_2.shape
      target_width = max(width1, width2)
      target_height = max(height1, height2)
      img_original_1 = Image.fromarray(img_original_1)
      img_original_2 = Image.fromarray(img_original_2)
      img_original_1 = img_original_1.resize((target_width, target_height))
      img_original_2 = img_original_2.resize((target_width, target_height))
      img_original_1 = np.asarray(img_original_1)
      img_original_2 = np.asarray(img_original_2)

      image_blended = cv.addWeighted(img_original_1, alpha, img_original_2, beta, gamma)
  
      # Plot
      if show_results:
        plt.subplot(131),
        plt.imshow(img_original_1)
        plt.title('First image')
        plt.subplot(132),
        plt.imshow(img_original_2)
        plt.title('Second image')
        plt.subplot(133)
        plt.imshow(image_blended)
        plt.title('Blended image')
        plt.show()

      return image_blended


    def find_contours(self, img_str, retrieval_mode='RETRIEVAL TREE', contour_approximation='CHAINING APPROXIMATION NONE', \
                      threshold_val=127, substitute_val=255, show_results=True):
      '''Find contours in an image after converting it from RGB to grayscale and applying a threshold.

      Inputs
      ------
      img_str:                        Local path to image as string / String
      retrieval_mode:                 Retrieval mode (default: 'RETRIEVAL TREE') / 1:
                                      'RETRIEVAL TREE': Retrieves all of the contours and reconstructs a full hierarchy of nested contours.
                                      'RETRIEVAL 2LEVEL': Retrieves all of the contours and organizes them into a two-level hierarchy.
                                      'RETRIEVAL LIST': Retrieves all of the contours without establishing any hierarchical relationships.
                                      'RETRIEVAL EXTERNAL': Retrieves only the extreme outer contours.
      contour_approximation:          Contour approximation (default: 'CHAINING APPROXIMATION NONE') / 1:
                                      'CHAINING APPROXIMATION NONE': All boundary points are stored
                                      'CHAINING APPROXIMATION SIMPLE': Removes redundant points
      threshold_val:                  Threshold value (default: 127) / 1
      substitute_val:                 Substitute value (default: 255) / 1
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      contours:                       Contours / Array
      Hierarchy                       Hierarchy / Array
      '''

      img_original = cv.imread(img_str)

      imgray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
      _, thresh = cv.threshold(imgray, threshold_val, substitute_val, 0)

      if retrieval_mode == 'RETRIEVAL TREE':
        _retrieval_mode = cv.RETR_TREE
      elif retrieval_mode == 'RETRIEVAL 2LEVEL':
        _retrieval_mode = cv.RETR_CCOMP
      elif retrieval_mode == 'RETRIEVAL LIST':
        _retrieval_mode = cv.RETR_LIST
      elif retrieval_mode == 'RETRIEVAL EXTERNAL':
        _retrieval_mode = cv.RETR_EXTERNAL
      else:
        raise TypeError("Retrieval mode unknown.")
      
      if contour_approximation == 'CHAINING APPROXIMATION NONE':
        _contour_approximation =  cv.CHAIN_APPROX_NONE
      elif contour_approximation == 'CHAINING APPROXIMATION SIMPLE':
        _contour_approximation = cv.CHAIN_APPROX_SIMPLE
      else:
        raise TypeError("Chain approximation mode unknown.")

      contours, hierarchy = cv.findContours(thresh, _retrieval_mode, _contour_approximation)

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        plt.imshow(cv.drawContours(img_original, contours, -1, (0, 255, 0), 3))
        plt.title('Contours')
        
        plt.show()
        
      return contours, hierarchy
    

    def get_contours_features_properties(self, img_str, retrieval_mode='RETRIEVAL TREE', contour_approximation='CHAINING APPROXIMATION NONE', \
                      threshold_val=127, substitute_val=255, show_results=True):
      ''' Get contour features and properties.

      Inputs
      ------
      contours:                       Contours determined by the function 'find_contours' / String

      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      moment:                       Moment / 1
      area                          Area / 1
      perimeter:                    Perimeter / 1
      hull:                         Hull (buldged out shape) / 1
      convexity:                    Convexity / 1
      bound_rect_unrotated:         Boundary rectangle unrotated [x, y, w, h] / 1
      bound_rect_rotated:           Boundary rectangle rotated / 1
      enclosing_circle:             Enclosing circle (x, y, radius) / 1
      enclosing_ellipse:            Enclosing ellipse / 1
      line:                         Fitted line / 1
      aspect_ratio:                 Aspect ration / 1
      rect_area:                    Area of rectangular / 1
      extent:                       Extent / 1
      solidity:                     Solidity / 1
      equivalent_diameter:          Equivalent diameter / 1
      pixelpoints:                  Points comprising the object / 1
      min_max_val_loc:              Minimum and maximum value and corresponding location 1
      mean_val:                     Mean color / 1
      extreme_points:               Extreme points / 1
      defects:                      Defects / 1
      '''

      img_original = cv.imread(img_str)

      imgray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
      _, thresh = cv.threshold(imgray, threshold_val, substitute_val, 0)

      if retrieval_mode == 'RETRIEVAL TREE':
        _retrieval_mode = cv.RETR_TREE
      elif retrieval_mode == 'RETRIEVAL 2LEVEL':
        _retrieval_mode = cv.RETR_CCOMP
      elif retrieval_mode == 'RETRIEVAL LIST':
        _retrieval_mode = cv.RETR_LIST
      elif retrieval_mode == 'RETRIEVAL EXTERNAL':
        _retrieval_mode = cv.RETR_EXTERNAL
      else:
        raise TypeError("Retrieval mode unknown.")
      
      if contour_approximation == 'CHAINING APPROXIMATION NONE':
        _contour_approximation =  cv.CHAIN_APPROX_NONE
      elif contour_approximation == 'CHAINING APPROXIMATION SIMPLE':
        _contour_approximation = cv.CHAIN_APPROX_SIMPLE
      else:
        raise TypeError("Chain approximation mode unknown.")

      contours, hierarchy = cv.findContours(thresh, _retrieval_mode, _contour_approximation)

      # Get moments, contour area, perimeter, convex hull, convexity, bounded rectangle, bounded rotated rectangle, 
      # enclosing circle, enclosing ellipse and fitted line
      moment = cv.moments(contours)
      area = cv.contourArea(contours)
      perimeter = cv.arcLength(contours, True)
      hull = cv.convexHull(contours)
      convexity = cv.isContourConvex(contours)
      _x_bound_rect, _y_bound_rect, _w_bound_rect, _h_bound_rect = cv.boundingRect(contours)
      bound_rect_unrotated = [_x_bound_rect, _y_bound_rect, _w_bound_rect, _h_bound_rect]
      bound_rect_rotated = cv.minAreaRect(contours)
      (_x_encl_circle, _y_encl_circle), _radius = cv.minEnclosingCircle(contours)
      enclosing_circle = [_x_encl_circle, _y_encl_circle, _radius]
      enclosing_ellipse = cv.fitEllipse(contours)
      _rows, _cols = img_original.shape[:2]
      [_vx_line, _vy_line, _x_line, _y_line] = cv.fitLine(contours, cv.DIST_L2, 0, 0.01, 0.01)
      line = [_vx_line, _vy_line, _x_line, _y_line]
      aspect_ratio = float(_w_bound_rect)/_h_bound_rect
      rect_area = _w_bound_rect*_h_bound_rect
      extent = float(area)/rect_area
      solidity = float(area)/cv.contourArea(hull)
      equivalent_diameter = np.sqrt(4*area/np.pi)
      mask = np.zeros(imgray.shape,np.uint8)
      pixelpoints = np.transpose(np.nonzero(mask))
      _min_val, _max_val, _min_loc, _max_loc = cv.minMaxLoc(imgray, mask=mask)
      min_max_val_loc = [_min_val, _max_val, _min_loc, _max_loc]
      mean_val = cv.mean(img_original, mask=mask)
      _leftmost = tuple(contours[contours[:,:,0].argmin()][0])
      _rightmost = tuple(contours[contours[:,:,0].argmax()][0])
      _topmost = tuple(contours[contours[:,:,1].argmin()][0])
      _bottommost = tuple(contours[contours[:,:,1].argmax()][0])
      extreme_points = [_leftmost, _rightmost, _topmost, _bottommost]
      defects = cv.convexityDefects(contours, hull)

      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img_original)
        plt.title('Original image')
        plt.subplot(122)
        cv.rectangle(img_original,(_x_bound_rect, _y_bound_rect),(_x_bound_rect+_w_bound_rect, _y_bound_rect+_h_bound_rect),(0,255,0),2)
        box = cv.boxPoints(bound_rect_rotated)
        box = np.int0(box)
        cv.drawContours(img_original,[box], 0, (0, 0, 255), 2)
        center = (int(_x_encl_circle), int(_y_encl_circle))
        radius = int(radius)
        cv.circle(img_original, center, radius, (0,255,0), 2)
        cv.ellipse(img_original, enclosing_ellipse, (255, 0, 0), 2)
        lefty = int((-_x_line*_vy_line/_vx_line) + _y_line)
        righty = int(((_cols-_x_line)*_vy_line/_vx_line) + _y_line)
        cv.line(img_original, (_cols-1, righty), (0, lefty), (0, 255, 0), 2)
        plt.title('Contours')
        
        plt.show()
        
      return (moment, area, perimeter, hull, convexity, bound_rect_unrotated, bound_rect_rotated, 
              enclosing_circle, enclosing_ellipse, line, aspect_ratio, extent, solidity, equivalent_diameter, 
              pixelpoints, min_max_val_loc, mean_val, extreme_points, defects)


    def shortest_distance_point_contour(self, img_str, point=(50, 50), retrieval_mode='RETRIEVAL TREE', contour_approximation='CHAINING APPROXIMATION NONE', \
                                        threshold_val=127, substitute_val=255, show_results=True):
      '''Get shortest distance of point to contour.

      Inputs
      ------
      img_str:                        Local path to image as string / String
      point:                          Point to measure the distance from to the contour (default: (50, 50)) / Tupel of float 
      retrieval_mode:                 Retrieval mode (default: 'RETRIEVAL TREE') / 1:
                                      'RETRIEVAL TREE': Retrieves all of the contours and reconstructs a full hierarchy of nested contours.
                                      'RETRIEVAL 2LEVEL': Retrieves all of the contours and organizes them into a two-level hierarchy.
                                      'RETRIEVAL LIST': Retrieves all of the contours without establishing any hierarchical relationships.
                                      'RETRIEVAL EXTERNAL': Retrieves only the extreme outer contours.
      contour_approximation:          Contour approximation (default: 'CHAINING APPROXIMATION NONE') / 1:
                                      'CHAINING APPROXIMATION NONE': All boundary points are stored
                                      'CHAINING APPROXIMATION SIMPLE': Removes redundant points
      threshold_val:                  Threshold value (default: 127) / 1
      substitute_val:                 Substitute value (default: 255) / 1
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      distance_shortest:            Shortest distance of point to contour / 1
      '''

      img_original = cv.imread(img_str)

      imgray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
      _, thresh = cv.threshold(imgray, threshold_val, substitute_val, 0)

      if retrieval_mode == 'RETRIEVAL TREE':
        _retrieval_mode = cv.RETR_TREE
      elif retrieval_mode == 'RETRIEVAL 2LEVEL':
        _retrieval_mode = cv.RETR_CCOMP
      elif retrieval_mode == 'RETRIEVAL LIST':
        _retrieval_mode = cv.RETR_LIST
      elif retrieval_mode == 'RETRIEVAL EXTERNAL':
        _retrieval_mode = cv.RETR_EXTERNAL
      else:
        raise TypeError("Retrieval mode unknown.")
      
      if contour_approximation == 'CHAINING APPROXIMATION NONE':
        _contour_approximation =  cv.CHAIN_APPROX_NONE
      elif contour_approximation == 'CHAINING APPROXIMATION SIMPLE':
        _contour_approximation = cv.CHAIN_APPROX_SIMPLE
      else:
        raise TypeError("Chain approximation mode unknown.")

      contours, hierarchy = cv.findContours(thresh, _retrieval_mode, _contour_approximation)

      distance_shortest = cv.pointPolygonTest(contours, point,True)

      return distance_shortest


    def match_shapes(self, img_str_1, img_str_2):
      '''Match shapes of the two given images and returns similarity metric.
      
      Inputs
      ------
      img_str_1:                      Local path to first image as string / String
      img_str_2:                      Local path to second image as string / String

      Outputs
      -------
      similarity:                     Similarity index / 1
      '''

      img_original_1 = cv.imread(img_str_1)
      img_original_2 = cv.imread(img_str_2)

      _, thresh = cv.threshold(img_str_1, 127, 255,0)
      _, thresh2 = cv.threshold(img_str_2, 127, 255,0)
      contours,hierarchy = cv.findContours(thresh,2,1)
      cnt1 = contours[0]
      contours,hierarchy = cv.findContours(thresh2,2,1)
      cnt2 = contours[0]
      
      similarity = cv.matchShapes(cnt1, cnt2, 1, 0.0)

      return similarity


    def get_histograms_1d(self, img_str, mask=None, show_results=True):
      ''' Determine histograms for given image in color or grayscale.

      Inputs
      ------
      img_str:                      Local path to image as string / String
      mask:                         Numpy array with the size of the image, where the 0-elements 
                                    are dropped and the 255-elements are kept (default: None) / 1
      show_results:                 Show results (default: True) / True; False

      Outputs
      -------
      histograms:                   Histograms (1x for grayscale image, 3x for color)
      '''

      img_original = cv.imread(img_str)
      n_channels = len(img_original.shape)
      histograms = []
      if n_channels==1:
        histograms = cv.calcHist([img_original], [0], mask, [256], [0, 256])
      elif n_channels==3:
        for i in range(n_channels):
          histograms.append(cv.calcHist([img_original], [i], mask, [256], [0, 256]))
      else:
        raise TypeError('Color space not supported (yet).')

      # Plot
      if show_results:
        color = ('b','g','r')
        plt.subplot(121)
        plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
        plt.title('Original image')
        plt.subplot(122)
        for i,col in enumerate(color):
            histr = cv.calcHist([img_original], [i], mask, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title('Histogram')
        plt.xlabel('Pixel value')
        plt.ylabel('Counts')
        plt.show()

      return histograms


    def equalize_histogram(self, img_str, mode='WHOLE', clip_limit=2.0, tile_grid_size=(8, 8), show_results=True):
      '''Apply histogram normalization to image after converting it to grayscale based on specific mode.

      Inputs
      ------
      img_str:                      Local path to image as string / String
      mode:                         Mode of histogram normalization (default: 'WHOLE'):
                                    'WHOLE': Equalization done over the entire image
                                    'TILES': Equalization done over image tiles using CLAHE method (Contrast Limited 
                                    Adaptive Histogram Equalization)
      clip_limit:                   Threshold for contrast limiting (default: 2.0) / 1.0
      tile_grid_size:               Size of grid for histogram equalization (default: (8, 8)) / 1
      show_results:                 Show results (default: True) / True; False

      Outputs
      -------
      image_equalized:              Equalized image using histogram
      '''

      img_original = cv.imread(img_str, cv.IMREAD_GRAYSCALE)

      if mode == 'WHOLE':
        image_equalized = cv.equalizeHist(img_original)
      elif mode == 'TILES':
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        image_equalized = clahe.apply(img_original)
      else:
        raise TypeError('Mode unkown.')

      # Plot
      if show_results:
        plt.subplot(131),
        plt.imshow(img_original, cmap='gray')
        plt.title('Original image')
        plt.subplot(132),
        plt.imshow(image_equalized, cmap='gray')
        plt.title('Equalized image')
        plt.show()

      return image_equalized


    def get_histogram_2d(self, img_str, mask=None, bins=[180, 256], range=[0, 180, 0, 255], show_results=True):
      ''' Determine 2D histograms for given image on hue-saturation pane.

      Inputs
      ------
      img_str:                      Local path to image as string / String
      mask:                         Numpy array with the size of the image, where the 0-elements 
                                    are dropped and the 255-elements are kept (default: None) / 1
      bins:                         Bins for HS-channels (default: [180, 256]) as list / 1
      range:                        Range for HS-channels(default: [0, 180, 0, 255]) as list / 1
      show_results:                 Show results (default: True) / True; False

      Outputs
      -------
      histograms:                   Histogram
      '''
      img_original = cv.imread(img_str)

      hsv = cv.cvtColor(img_original,cv.COLOR_BGR2HSV)
      histograms = cv.calcHist([hsv], [0, 1], mask, bins, range)

      if show_results:
        plt.subplot(131),
        plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
        plt.title('Original image')
        plt.subplot(132)
        plt.imshow(histograms, interpolation='nearest')
        plt.title('2D histogram')
        plt.xlabel('Saturation (S)')
        plt.ylabel('Hue (H)')
        plt.show()

      return histograms
    

    def backproject_histogram(self, img_str_roi, img_str_target, mask=None, bins=[180, 256], range=[0, 180, 0, 255], show_results=True):
      ''' Determine 2D histograms for given image on hue-saturation pane.

      Inputs
      ------
      img_str_roi:                  Region of interest as local path to image / String
      img_str_target:               Target as local path to image / String
      mask:                         Numpy array with the size of the image, where the 0-elements 
                                    are dropped and the 255-elements are kept (default: None) / 1
      bins:                         Bins for HS-channels (default: [180, 256]) as list / 1
      range:                        Range for HS-channels(default: [0, 180, 0, 255]) as list / 1
      show_results:                 Show results (default: True) / True; False

      Outputs
      -------
      image_backprojected:          Backprojected image
      '''

      image_roi = cv.imread(img_str_roi)
      hsv = cv.cvtColor(image_roi, cv.COLOR_BGR2HSV)
      image_target = cv.imread(img_str_target)
      hsvt = cv.cvtColor(image_target, cv.COLOR_BGR2HSV)
      
      # Calculating object histogram
      roihist = cv.calcHist([hsv],[0, 1], None, bins, range)
      
      # Normalize histogram and apply backprojection
      cv.normalize(roihist,roihist, 0, 255, cv.NORM_MINMAX)
      dst = cv.calcBackProject([hsvt], [0,1], roihist, range, 1)
      
      # Now convolute with circular disc
      disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
      cv.filter2D(dst, -1, disc,dst)
      
      # Threshold and binary AND
      _, thresh = cv.threshold(dst, 50, 255, 0)
      thresh = cv.merge((thresh, thresh, thresh))
      image_backprojected = cv.bitwise_and(image_target, thresh)
      
      if show_results:
        plt.subplot(131)
        plt.imshow(image_target)
        plt.title('Target image')
        plt.subplot(132)
        plt.imshow(image_roi)
        plt.title('ROI image')
        plt.subplot(133)
        plt.imshow(image_backprojected)
        plt.title('Backprojected image')
        plt.show()

      return image_backprojected


    def filter_fourier(self, img_str, type='HIGH PASS', size_lpf=30, show_results=True):
      ''' Filter image based on selected filter type

      Inputs
      ------
      img_str:                  Local path to image as string / String
      type:                     Type of Fourier filter as string (default: 'HIGH PASS'):
                                  'HIGH PASS': High-pass filter using DFT
                                  'LOW PASS': Low-pass filter using DFT
      size_lpf:                 Size of quadratic kernel for low-pass filtering (default: 30) / px                            
      show_results:             Show results (default: True) / True; False

      Outputs
      -------
      image_filtered:           Filtered image
      '''

      img_original = cv.imread(img_str, cv.IMREAD_GRAYSCALE)

      if type == 'HIGH PASS':
        dft = cv.dft(np.float32(img_original), flags = cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        image_filtered = 20*np.log(cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
      elif type == 'LOW PASS':
        rows, cols = img_original.shape
        crow, ccol = rows//2, cols//2
        mask = np.zeros((rows,cols,2),np.uint8)
        mask[crow-size_lpf:crow+size_lpf, ccol-size_lpf:ccol+size_lpf] = 1
        dft = cv.dft(np.float32(img_original), flags = cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv.idft(f_ishift)
        image_filtered = cv.magnitude(img_back[:, :, 0],img_back[:, :, 1])
      else:
        raise TypeError('Type unknown.')

      # Plot
      if show_results:
        plt.subplot(131),
        plt.imshow(img_original, cmap='gray')
        plt.title('Original image')
        plt.subplot(132),
        plt.imshow(image_filtered, cmap='gray')
        if type == 'HIGH PASS':
          plt.title('High-pass filtered image')
        elif type == 'LOW PASS':
          plt.title('Low-pass filtered image')
        plt.show()

      return image_filtered


    def template_matching(self, img_str, template_str, method='TM_CCOEFF', show_results=True):
      ''' Find pattern in image based on specific method. Multiple occurrences are detected.

      Inputs
      ------
      img_str:                  Local path to image as string / String
      template_str:             Local path to template image as string / String
      method:                   Method of template matching as string (default: 'TM_CCOEFF'):
                                  'TM_CCOEFF': Correlation coefficient
                                  'TM_CCOEFF_NORMED': Normalized correlation coefficient
                                  'TM_CCORR': Cross-correlation
                                  'TM_CCORR_NORMED': Normalized cross-correlation
                                  'TM_SQDIFF':  Sum of squared difference
                                  'TM_SQDIFF_NORMED': Normalized sum of squared difference                        
      show_results:             Show results (default: True) / True; False

      Outputs
      -------
      locations:                Locations where template was detected.
      '''

      img_original = cv.imread(img_str)
      img_original_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
      img_template = cv.imread(template_str, cv.IMREAD_GRAYSCALE)

      w, h = img_template.shape[::-1]
      _method = getattr(cv, method)
      res = cv.matchTemplate(img_original_gray, img_template, _method)
      threshold = 0.8
      locations = np.where( res >= threshold)

      if show_results:
        for pt in zip(*locations[::-1]):
            cv.rectangle(img_original, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
        plt.title(f'Original image {method}')
        plt.show()

      return locations


    def find_hough_lines(self, img_str, canny_thd_1=50, canny_thd_2=150, sobel_size=3, rho_res=1, theta_res=np.pi/180, threshold=200, show_results=True):
      ''' Find lines using Hough transformation. For this, all pixels with regard to the parameters (d: distance, alpha: angle; 
      both referenced to the origin of the image) are upvoted that fullfill the specific parameter sets. Using a threshold,
      the maximum of votes are determined that deliver the parameter sets with highly probability of representing a line.

      Inputs
      ------
      img_str:                  Local path to image as string / String
      canny_thd_1:              First threshold of the hysteresis procedure of canny edge detection (default: 50) / 1
      canny_thd_2:              Second threshold of the hysteresis procedure of canny edge detection (default: 150) / 1
      sobel_size:               Aperture size of Sobel operator (default: 3) / 1
      rho_res:                  Distance resolution (default: 1) / 1
      theta_res:                Angle resolution (default: np.pi/180) / rad
      threshold:                Threshold (default: 200) / 1
      show_results:             Show results (default: True) / True; False

      Outputs
      -------
      locations:                Locations where template was detected.
      '''

      img_original = cv.imread(img_str)
      gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
      edges = cv.Canny(gray, canny_thd_1, canny_thd_2, apertureSize=sobel_size)
      lines = cv.HoughLines(edges, rho_res, theta_res, threshold)

      if show_results:
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(img_original, (x1, y1), (x2, y2), (0, 0, 255), 2)

        plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
        plt.title(f'Original image')
        plt.show()

      return lines


    def find_hough_circle(self, img_str, blur_kernel_size=5, method='HOUGH GRADIENT', accumulator_resolution=1, min_distance=20, param_1=50, param_2=30, min_radius=0, max_radius=0, show_results=True):
      ''' Find circle using Hough Transformation, which uses gradients.

      Inputs
      ------
      img_str:                  Local path to image as string / String
      blur_kernel_size:         Size of blurring kernel (default: 5) / 1
      method:                   Method of Hough Transformation for circle detection as string:
                                  'HOUGH GRADIENT': Standard Hough gradient
                                  'HOUGH GRADIENT ALT': Alternative Hough gradient
      accumulator_resolution:   Inverse resolution of accumulator. For 1, the accumulator has the same size as the image.
      min_distance:             Minimum distance between circle centers
      param_1:                  First method-specific parameter (read reference of cv.HoughCircles; default: 50) / 1
      param_2:                  Second method-specific parameter (read reference of cv.HoughCircles; default: 30) / 1
      min_radius:               Minimum radius of circles to detect (read reference of cv.HoughCircles; default: 0) / 1
      max_radius:               Maximum radius of circles to detect (read reference of cv.HoughCircles; default: 0) / 1
      show_results:             Show results (default: True) / True; False

      Outputs
      -------
      locations:                Locations where template was detected.
      '''

      img_original = cv.imread(img_str, cv.IMREAD_GRAYSCALE)
      img_original = cv.medianBlur(img_original, blur_kernel_size)
      cimg = cv.cvtColor(img_original, cv.COLOR_GRAY2BGR)
      
      if method == 'HOUGH GRADIENT':
        _method = cv.HOUGH_GRADIENT
      elif method == 'HOUGH GRADIENT ALT':
        _method = cv.HOUGH_GRADIENT_ALT
      else:
        raise TypeError('Method unknown.')

      circles = cv.HoughCircles(img_original, _method, accumulator_resolution, min_distance, param1=param_1, param2=param_2, minRadius=min_radius, maxRadius=max_radius)
      circles = np.uint16(np.around(circles))

      if show_results:
        for i in circles[0,:]:
          cv.circle(cimg,(i[0], i[1]), i[2], (0, 255, 0),2)
          cv.circle(cimg,(i[0], i[1]), 2,(0, 0, 255),3)

        plt.imshow(cv.cvtColor(cimg, cv.COLOR_BGR2RGB))
        plt.title(f'Original image')
        plt.show()

      return circles


    def segment_watershed(self, img_str, size_kernel_opening_dilation=3, n_iter_opening=2, n_iter_dilation=3, mask_size_distance_transform=5, threshold_coeff=0.7, show_results=True):
      ''' Apply segmentation using watershed algorithm

      Inputs
      ------
      img_str:                        Local path to image as string / String
      size_kernel_opening_dilation:   Size of quadratic kernel for opening (default: 3) / 1
      n_iter_opening:                 Number of iterations for opening (default: 2) / 1
      n_iter_dilation:                Number of iterations for dilation (default: 3) / 1
      mask_size_distance_transform:   Mask size of distance transform (default: 5) / 1
      threshold_coeff:                Threshold coefficient multiplied with maximum value (default: 0.7) / 1.0
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      img_original:                   Image of segmented elements
      '''

      img_original = cv.imread(img_str)
      gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

      # Apply threshold to create an estimate of the coin regions
      ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

      # Apply noise removal based on opening
      kernel = np.ones((size_kernel_opening_dilation, size_kernel_opening_dilation), np.uint8)
      opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=n_iter_opening)
      
      # Apply dilation to get sure background area
      sure_bg = cv.dilate(opening, kernel, iterations=n_iter_dilation)
      
      # Finding sure foreground area
      dist_transform = cv.distanceTransform(opening, cv.DIST_L2, mask_size_distance_transform)
      ret, sure_fg = cv.threshold(dist_transform, threshold_coeff*dist_transform.max(), 255, 0)
      
      # Finding unknown region
      sure_fg = np.uint8(sure_fg)
      unknown = cv.subtract(sure_bg,sure_fg)

      # Marker labelling
      ret, markers = cv.connectedComponents(sure_fg)
      
      # Add one to all labels so that sure background is not 0, but 1
      markers = markers+1
      
      # Now, mark the region of unknown with zero
      markers[unknown==255] = 0
      markers = cv.watershed(img_original, markers)
      img_original[markers == -1] = [255, 0, 0]

      if show_results:
        plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
        plt.title(f'Original image')
        plt.show()

      return img_original


    def segment_grabcut(self, img_str, rect_selection_foreground=(50, 50, 450, 290), n_iter=5, show_results=True):
      ''' Apply segmentation using watershed algorithm initialized with rectangle.

      Inputs
      ------
      img_str:                        Local path to image as string / String
      rect_selection_foreground:      Rectangle for selecting sure foreground as 4-tupel (x, y, w, h) (default: (50, 50, 450, 290))/ Tupel
      n_iter:                         Number of iterations of GrabCut-algorithm (default: 5) / 1
      show_results:                   Show results (default: True) / True; False

      Outputs
      -------
      img_segmented:                  Segmented image
      '''

      img = cv.imread(img_str)
      assert img is not None, "file could not be read, check with os.path.exists()"
      mask = np.zeros(img.shape[:2],np.uint8)
      
      bgdModel = np.zeros((1,65),np.float64)
      fgdModel = np.zeros((1,65),np.float64)
      rect = rect_selection_foreground

      cv.grabCut(img,mask,rect,bgdModel,fgdModel, n_iter, cv.GC_INIT_WITH_RECT)
      
      mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
      img_segmented = img*mask2[:, :, np.newaxis]
      
      if show_results:
        plt.imshow(img_segmented), plt.colorbar()
        plt.title('Segmented image')
        plt.show()

      return img_segmented


    # Feature extraction
    #-------------------

    def canny_edge_detection(self, img_str, show_results=True):
      '''Apply Canny Edge detector to picture example.

      Inputs
      ------
      img_str:          Local path to image as string / String

      Outputs
      ------
      edges:            Edges / 1
      '''

      img = cv.imread(img_str, cv.IMREAD_GRAYSCALE)
      assert img is not None, "file could not be read, check with os.path.exists()"
      edges = cv.Canny(img, 100, 200)
      
      # Plot
      if show_results:
        plt.subplot(121),
        plt.imshow(img,cmap='gray')
        plt.title('Original Image')
        plt.subplot(122)
        plt.imshow(edges,cmap='gray')
        plt.title('Edge Image')
        
        plt.show()

      return edges


    def harris_corner_detector(self, img_str, show_results=True):
      '''Apply Harris Corner detector to picture example.

      Inputs
      ------
      img_str:          Local path to image / String
      show_results:     Show results (Default: True) / True; False

      Outputs:
      --------
      img:              Processed image / Image
      '''

      img = cv.imread(img_str)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      gray = np.float32(gray)

      # Apply Harris Corner detection
      dst = cv.cornerHarris(gray, 2, 3, 0.04)
      
      # result is dilated for marking the corners, not important
      dst = cv.dilate(dst, None)
      
      # Threshold for an optimal value, it may vary depending on the image.
      img[dst > 0.01*dst.max()] = [0, 0, 255]
      
      # Plot
      if show_results:
        cv.imshow('dst', img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

      return img


    def harris_corner_detector_subpixel_accuracy(self, img_str, show_results=True):
      '''Apply Harris Corner detector with subpixel accuracy to picture example.

      Inputs
      ------
      img_str:          Local path to image / String
      show_results:     Show results (Default: True) / True; False

      Outputs:
      --------
      img:              Processed image / Image
      '''

      img = cv.imread(img_str)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      
      # Find Harris corners
      gray = np.float32(gray)
      dst = cv.cornerHarris(gray, 2, 3, 0.04)
      dst = cv.dilate(dst,None)
      ret, dst = cv.threshold(dst, 0.01*dst.max(), 255, 0)
      dst = np.uint8(dst)
      
      # Find centroids
      ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
      
      # Define the criteria to stop and refine the corners
      criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
      corners = cv.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
      
      # Plot
      res = np.hstack((centroids, corners))
      res = np.int0(res)
      img[res[:,1],res[:,0]]=[0,0,255]
      img[res[:,3],res[:,2]] = [0,255,0]
      
      if show_results:
        cv.imshow('dst', img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

      return img


    def shi_tomasi_corner_detector(self, img_str, max_corners=25, quality_level=0.01, min_distance=10, show_results=True):
      '''Apply Shi-Tomasi corner detector.

      Inputs
      ------
      img_str:          Local path to image / String
      max_corners:      Maximum number of corners (default: 25) / 1
      quality_level:    Quality level (default: 0.01). Quality times quality level defines the threshold. / 0.01
      min_distance:     Minimum distance between corners / 1
      show_results:     Show results (Default: True) / True; False

      Outputs:
      --------
      corners:          Detected corners
      '''

      img = cv.imread(img_str)
      gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
      
      corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
      corners = np.int0(corners)
      
      if show_results:
        for i in corners:
            x,y = i.ravel()
            cv.circle(img, (x,y), 3, 255, -1)

        plt.imshow(img)
        plt.title('Original image')
        plt.show()

      return corners


    def sift_feature_detector(self, img_str, show_results=True):
      '''Apply Scale-Invariant Feature Transform (SIFT) algorithm to detect features that are described by 
      keypoints and corresponding descriptors.

      Inputs
      ------
      img_str:          Local path to image / String
      show_results:     Show results (Default: True) / True; False

      Outputs:
      --------
      keypoints:          Keypoints of detected corners
      descriptors:        Descriptors of detected corners
      '''

      img_original = cv.imread(img_str)
      img_gray_original = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
      
      sift = cv.SIFT_create()
      keypoints, descriptors = sift.detectAndCompute(img_gray_original, None)
      
      img = cv.drawKeypoints(img_gray_original, keypoints, img_original, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

      if show_results:
        plt.imshow(img)
        plt.title('Original image')
        plt.show()
      
      return keypoints, descriptors


    def surf_feature_detector(self, img_str, hessian_threshold=400, upright=True, show_results=True):
      '''[THIS FUNCTION IS NOT WORKING SINCE OPENCV HAS PATENT ISSUES]
      Apply Speeded-Up Robust Features (SURF) algorithm to detect features that are described by 
      keypoints and corresponding descriptors.

      Inputs
      ------
      img_str:            Local path to image / String
      hessian_threshold:  Hessian threshold (default: 400) / 1
      upright:            Upright flag / True; False
      show_results:       Show results (Default: True) / True; False

      Outputs:
      --------
      keypoints:          Keypoints of detected corners
      descriptors:        Descriptors of detected corners
      '''

      img_original = cv.imread(img_str, cv.IMREAD_GRAYSCALE)
      surf = cv.xfeatures2d.SURF_create(hessian_threshold)
      
      if upright:
        surf.setUpright(True)
      else:
        surf.setUpright(False)
    
      keypoints, descriptors = surf.detectAndCompute(img_original, None)

      if show_results:
        img_plot = cv.drawKeypoints(img_original, keypoints, None, (255, 0, 0), 4)
        plt.imshow(img_plot)
        plt.title('Original image')
        plt.show()

      return keypoints, descriptors
    

    def fast_feature_detector(self, img_str, threshold=10, nonmax_suppresion=True, type='9 16', show_results=True):
      '''Apply Features from Accelerated Segment Test (FAST) algorithm to detect features that are described 
      by keypoints and corresponding descriptors.

      Inputs
      ------
      img_str:            Local path to image / String
      threshold:          Threshold of feature detection (default: 10) / 1
      nonmax_suppresion:  Non-maximum suppression (default: True) / True; False
      type:               Type of feature detector (default: cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
                          '5 8': cv.FAST_FEATURE_DETECTOR_TYPE_5_8
                          '7 12': cv.FAST_FEATURE_DETECTOR_TYPE_7_12
                          '9 16': cv.FAST_FEATURE_DETECTOR_TYPE_9_16
      show_results:       Show results (Default: True) / True; False

      Outputs:
      --------
      keypoints:          Keypoints of detected corners
      '''

      img_original = cv.imread(img_str, cv.IMREAD_GRAYSCALE)
      
      fast = cv.FastFeatureDetector_create()
      if nonmax_suppresion:
        fast.setNonmaxSuppression(1)
      else:
        fast.setNonmaxSuppression(0)

      fast.setThreshold(threshold)
      if type == '5 8':
        _type = cv.FastFeatureDetector_TYPE_5_8
      elif type == '7 12':
        _type = cv.FastFeatureDetector_TYPE_7_12
      elif type == '9 16':
        _type = cv.FastFeatureDetector_TYPE_9_16
      fast.setType(_type)

      keypoints = fast.detect(img_original, None)
      img_keypoints = cv.drawKeypoints(img_original, keypoints, None, color=(255, 0, 0))
      
      if show_results:
        plt.imshow(img_keypoints)
        plt.title('Original image with keypoints')
        plt.show()
      
      return keypoints


    def censure_with_brief(self, img_str, show_results=True):
      '''Apply CenSurE (STAR) feature detector using BRIEF feature descriptor.

      Inputs
      ------
      img_str:            Local path to image / String
      show_results:       Show results (Default: True) / True; False

      Outputs:
      --------
      keypoints:          Keypoints of detected corners
      descriptors:        Descriptors of detected corners
      '''

      img_original = cv.imread(img_str, cv.IMREAD_GRAYSCALE)
      
      # Initiate FAST detector
      star = cv.xfeatures2d.StarDetector_create()
      
      # Initiate BRIEF extractor
      brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
      
      # find the keypoints with STAR
      keypoints = star.detect(img_original, None)

      # compute the descriptors with BRIEF
      keypoints, descriptors = brief.compute(img_original, keypoints)

      if show_results:
        img_keypoints = cv.drawKeypoints(img_original, keypoints, img_original, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img_keypoints)
        plt.title('Original image with keypoints')
        plt.show()

      return keypoints, descriptors


    def orb_feature_detector(self, img_str, n_features=500, n_points_per_element=2, score_type='HARRIS', show_results=True):
      '''Apply ORB algorithm feature detector which combines the FAST keypoint detector using BRIEF feature descriptor. 
      The most relevant parameters can be set.

      Inputs
      ------
      img_str:                Local path to image / String
      n_features:             Maximum number of features to retain (default: 500) / 1
      n_points_per_element:   Number of points that produce each element of the oriented BRIEF descriptor (default: 2) / 1
      score_type:             Score type (default: 'HARRIS')
                                'HARRIS': cv.ORB_HARRIS_SCORE
                                'FAST': cv.ORB_FAST_SCORE
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      keypoints:              Keypoints of detected corners
      descriptors:            Descriptors of detected corners
      '''
    
      img_original = cv.imread(img_str, cv.IMREAD_GRAYSCALE)

      # Initiate ORB detector
      if score_type == 'HARRIS':
        _score_type = cv.ORB_HARRIS_SCORE
      elif score_type == 'FAST':
        _score_type = cv.ORB_FAST_SCORE
      else:
        raise TypeError('Unknown score type.')
      orb = cv.ORB_create(nfeatures=n_features, scoreType=_score_type, WTA_K=n_points_per_element)
      
      # find the keypoints with ORB
      keypoints = orb.detect(img_original, None)
      
      # compute the descriptors with ORB
      keypoints, descriptors = orb.compute(img_original, keypoints)
      
      # draw only keypoints location,not size and orientation
      if show_results:
        img_plot = cv.drawKeypoints(img_original, keypoints, None, color=(0, 255, 0), flags=0)
        plt.imshow(img_plot)
        plt.title('Original image with keypoints')
        plt.show()

      return keypoints, descriptors


    def feature_matching(self, img_str_1, img_str_2, type='BRUTE FORCE WITH ORB', find_homography_flann=True, min_match_count=10, show_results=True):
      '''Apply feature matching algorithm based on type (predefined descriptors, matching type and related default parameters).

      Inputs
      ------
      img_str_1:              Local path to first image / String
      img_str_2:              Local path to second image / String
      type:                   Type (default: 'BRUTE FORCE WITH SIFT')
                                'BRUTE FORCE WITH ORB': Brute-force matching with SIFT descriptors
                                'BRUTE FORCE WITH SIFT': Brute-force matching with ORB descriptors
                                'FLANN WITH SIFT': Fast Library for Approximate Nearest Neighbors (FLANN-)based matching with SIFT descriptors
      find_homography_flann:  Find homography when used with FLANN method (default: True) / True; False
      min_match_count:        Minimum number of matches (default: 10) / 1
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      matches:                List of matches
      '''

      img_query = cv.imread(img_str_1, cv.IMREAD_GRAYSCALE)
      img_train = cv.imread(img_str_2, cv.IMREAD_GRAYSCALE)

      if type == 'BRUTE FORCE WITH ORB':

        # Initiate ORB detector
        orb = cv.ORB_create()
        
        # Find the keypoints and descriptors with ORB
        keypoints_query, descriptors_query = orb.detectAndCompute(img_query, None)
        keypoints_train, descriptors_train = orb.detectAndCompute(img_train, None)

        # Create BFMatcher object and match decriptors
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_query, descriptors_train)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        # Draw first 10 matches
        if show_results:
          img_plot = cv.drawMatches(img_query ,keypoints_query, img_train, keypoints_train, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
          plt.imshow(img_plot)
          plt.title('Query and train image')
          plt.show()

      elif type == 'BRUTE FORCE WITH SIFT':

        # Initiate SIFT detector
        sift = cv.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        keypoints_query, descriptors_query = sift.detectAndCompute(img_query, None)
        keypoints_train, descriptors_train = sift.detectAndCompute(img_train, None)
        
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors_query, descriptors_train, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        # cv.drawMatchesKnn expects list of lists as matches.
        if show_results:
          img_plot = cv.drawMatchesKnn(img_query, keypoints_query, img_train, keypoints_train, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
          plt.imshow(img_plot)
          plt.title('Query and train image')
          plt.show()

      elif type == 'FLANN WITH SIFT':

        # Initiate SIFT detector
        sift = cv.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        keypoints_query, descriptors_query = sift.detectAndCompute(img_query, None)
        keypoints_train, descriptors_train = sift.detectAndCompute(img_train, None)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descriptors_query, descriptors_train, k=2)
        
        if not find_homography_flann:

          # Need to draw only good matches, so create a mask
          matches_mask = [[0, 0] for i in range(len(matches))]
          
          # ratio test as per Lowe's paper
          for i,(m,n) in enumerate(matches):
              if m.distance < 0.7*n.distance:
                  matches_mask[i]=[1,0]
          
          draw_params = dict(matchColor = (0, 255, 0),
                            singlePointColor = (255 ,0, 0),
                            matchesMask = matches_mask,
                            flags = cv.DrawMatchesFlags_DEFAULT)
          
          if show_results:
            img_plot = cv.drawMatchesKnn(img_query, keypoints_query, img_train, keypoints_train, matches, None, **draw_params)
            plt.imshow(img_plot)
            plt.title('Query and train image')
            plt.show()

        else:

          # Store all the good matches as per Lowe's ratio test.
          good = []
          for m,n in matches:
              if m.distance < 0.7*n.distance:
                  good.append(m)

          if len(good) > min_match_count:
              src_pts = np.float32([ keypoints_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
              dst_pts = np.float32([ keypoints_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
          
              M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
              matches_mask = mask.ravel().tolist()
          
              h,w = img_query.shape
              pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
              dst = cv.perspectiveTransform(pts, M)
              img_train = cv.polylines(img_train,[np.int32(dst)], True, 255, 3, cv.LINE_AA)
          
          else:
              print( "Not enough matches are found - {}/{}".format(len(good), min_match_count) )
              matches_mask = None

          draw_params = dict(matchColor=(0, 255 ,0),   # Draw matches in green color
                            singlePointColor=None,
                            matchesMask=matches_mask,  # Draw only inliers
                            flags=2)
          
          if show_results:
            img_plot = cv.drawMatches(img_query, keypoints_query, img_train, keypoints_train, good, None, **draw_params)
            plt.imshow(img_plot, 'gray')
            plt.show()

      return keypoints_query, keypoints_train, descriptors_query, descriptors_train, matches

    # Camera calibration and 3D reconstruction
    #------------------------------------------

    def calibrate_camera(self, folder_img, img_str_sample, show_results=True):
      '''Calibrate camera given a sample image. After finding the camera matrix, distortion coefficients, rotation and translation vectors, an undistortion of
      the sample image is applied and the reprojection error is evaluated.

      Inputs
      ------
      folder_img:             Folder containing images to get calibration data
      img_str_sample:         Local path to sample image / String
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      mtx:                    Camera matrix
      dist:                   Distortion coefficients
      rvecs:                  Rotation vector
      tvecs:                  Translation vector
      '''

      # termination criteria
      criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
      
      # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
      objp = np.zeros((6*7, 3), np.float32)
      objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
      
      # Arrays to store object points and image points from all the images.
      objpoints = [] # 3d point in real world space
      imgpoints = [] # 2d points in image plane.
      
      images = os.listdir(folder_img)
      for i in range(len(images)):
        images[i] = folder_img + images[i]
      
      for fname in images:
          img = cv.imread(fname)
          gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      
          # Find the chess board corners
          ret, corners = cv.findChessboardCorners(gray, (7,6), None)
      
          # If found, add object points, image points (after refining them)
          if ret == True:
              objpoints.append(objp)
      
              corners2 = cv.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
              imgpoints.append(corners2)
      
              # Draw and display the corners
              if show_results:
                cv.drawChessboardCorners(img, (7, 6), corners2, ret)
                cv.imshow('Image of sample folder', img)
                cv.waitKey(500)

      # Apply calibration
      ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

      # Apply undistortion
      img = cv.imread(img_str_sample)
      h,  w = img.shape[:2]
      newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
      dst = cv.undistort(img, mtx, dist, None, newcameramtx)
      
      # Crop the image and plot if desired
      if show_results:
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        plt.imshow(dst)
        plt.title('Undistorted sample image')
        plt.show()

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(objpoints)) )

      return mtx, dist, rvecs, tvecs


    def calculate_pose(self, folder_img, img_str_sample, show_results=True):
      '''Calculate pose of camera based on the calibration parameters (camera matrix, distortion coefficients). For this, the function 
      'calibrate_camera' is called first.

      Inputs
      ------
      folder_img:             Folder containing images to get calibration data
      img_str_sample:         Local path to sample image / String
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      rvecs:                  Rotation vector
      tvecs:                  Translation vector
      '''

      #  Function to draw coordinate axes after projecting the lines from 3d to 2d
      def draw(img, corners, imgpts):
          corner = tuple(corners[0].ravel().astype(int))
          img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
          img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
          img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)
          return img

      # Call camera calibration matrix first. Discard rotation and translation vectors and determine them using a function to find 3D-2D point correspondences. 
      mtx, dist, _, _ = mv_1.calibrate_camera(folder_img=folder_img, img_str_sample=img_str_sample, show_results=False)

      criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
      objp = np.zeros((6*7,3), np.float32)
      objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1, 2)
      
      axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1,3)

      images = os.listdir(folder_img)
      for i in range(len(images)):
        images[i] = folder_img + images[i]

      for fname in images:
          img = cv.imread(fname)
          gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
          ret, corners = cv.findChessboardCorners(gray, (7, 6),None)
      
          if ret == True:
              corners2 = cv.cornerSubPix(gray,corners,(11, 11),(-1, -1), criteria)
      
              # Find the rotation and translation vectors.
              ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
      
              # project 3D points to image plane
              imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
      
              if show_results:
                img = draw(img, corners2, imgpts)
                cv.imshow('img', img)
                k = cv.waitKey(0) & 0xFF
                if k == ord('s'):
                    cv.imwrite(fname[:6]+'.png', img)

      return rvecs, tvecs


    def find_epilines(self, img_str_1, img_str_2, show_results=True):
      '''Find epilines using two images containing the same object but from different perspectives.

      Inputs
      ------
      img_str_1:              Local path to first image / String
      img_str_2:              Local path to second image / String
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      lines_1:                Epilines in the first image
      lines_2:                Epilines in the second image
      '''

      # Function to draw epilines
      def drawlines(img1,img2,lines,pts1,pts2):
          ''' img1 - image on which we draw the epilines for the points in img2
              lines - corresponding epilines '''
          r,c = img1.shape
          img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
          img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
          for r,pt1,pt2 in zip(lines,pts1,pts2):
              color = tuple(np.random.randint(0,255,3).tolist())
              x0,y0 = map(int, [0, -r[2]/r[1] ])
              x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
              img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
              img1 = cv.circle(img1,tuple(pt1),5,color,-1)
              img2 = cv.circle(img2,tuple(pt2),5,color,-1)
          return img1,img2

      # Get keypoints
      img_1 = cv.imread(img_str_1, cv.IMREAD_GRAYSCALE)
      img_2 = cv.imread(img_str_2, cv.IMREAD_GRAYSCALE)
      keypoints_query, keypoints_train, descriptors_query, descriptors_train, matches = self.feature_matching(img_str_1, img_str_2, type='FLANN WITH SIFT', find_homography_flann=False, min_match_count=10, show_results=False)

      # Apply ratio test as per Lowe's paper
      pts_1 = []
      pts_2 = []
      for i,(m,n) in enumerate(matches):
          if m.distance < 0.8*n.distance:
              pts_2.append(keypoints_train[m.trainIdx].pt)
              pts_1.append(keypoints_query[m.queryIdx].pt)

      # Find fundamental matrix and select only inlier points
      pts_1 = np.int32(pts_1)
      pts_2 = np.int32(pts_2)
      F, mask = cv.findFundamentalMat(pts_1, pts_2, cv.FM_LMEDS)
      pts_1 = pts_1[mask.ravel()==1]
      pts_2 = pts_2[mask.ravel()==1]

      # Find epilines corresponding to points in right image (second image) and
      # drawing its lines on left image
      lines_1 = cv.computeCorrespondEpilines(pts_2.reshape(-1, 1, 2), 2, F)
      lines_1 = lines_1.reshape(-1, 3)
      
      # Find epilines corresponding to points in left image (first image) and
      # drawing its lines on right image
      lines_2 = cv.computeCorrespondEpilines(pts_1.reshape(-1, 1, 2), 1, F)
      lines_2 = lines_2.reshape(-1, 3)
      
      if show_results:
        img_5, img_6 = drawlines(img_1, img_2, lines_1, pts_1, pts_2)
        img_3, img_4 = drawlines(img_2, img_1, lines_2, pts_2, pts_1)
        plt.subplot(121)
        plt.imshow(img_5)
        plt.title('Left image with epilines')
        plt.subplot(122)
        plt.imshow(img_3)
        plt.title('Right image with epilines')
        plt.show()

      return lines_1, lines_2


    def get_depth_map(self, img_str_1, img_str_2, num_disparities=16, block_size=15, show_results=True):
      '''Find depth map using two images containing the same object but from different perspectives.

      Inputs
      ------
      img_str_1:              Local path to first image / String
      img_str_2:              Local path to second image / String
      num_disparities:        Number of disparities (default: 16). The larger, the better but more computational effort is required. / 1
      block_size:             Block size (default: 15). Larger block size implies smoother, though less accurate disparity map. Smaller 
                              block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong 
                              correspondence / 1
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      disparity:              Disparity map
      '''

      img_left = cv.imread(img_str_1, cv.IMREAD_GRAYSCALE)
      img_right = cv.imread(img_str_2, cv.IMREAD_GRAYSCALE)
      
      stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
      disparity = stereo.compute(img_left, img_right)
      if show_results:
        plt.imshow(disparity, 'gray')
        plt.title('Disparity map')
        plt.show()

      return disparity


    # Computational photography
    #--------------------------

    def denoise_non_local_means(self, img_str, show_results=True):
      '''Denoise single image based on the non-local means algorithm.

      Inputs
      ------
      img_str:            Local path to image / String
      show_results:       Show results (Default: True) / True; False

      Outputs:
      --------
      keypoints:          Keypoints of detected corners
      img_denoised:       Denoised image
      '''

      img_original = cv.imread(img_str)

      if len(img_original.shape) == 1:
        img_denoised = cv.fastNlMeansDenoising(img_original, None, 10, 10, 7, 21)
      elif len(img_original.shape) == 3:
        img_denoised = cv.fastNlMeansDenoisingColored(img_original, None, 10, 10, 7, 21)
      else:
        raise TypeError('Unable to determine if image is color or gray-scale.')

      if show_results:
        plt.subplot(121)
        if len(img_original.shape) == 3:
          plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
        plt.title('Original image')
        plt.subplot(122)
        if len(img_original.shape) == 3:
          plt.imshow(cv.cvtColor(img_denoised, cv.COLOR_BGR2RGB))
        plt.title('Denoised image')
        plt.show()

      return img_denoised


    def inpaint(self, img_str, img_str_mask, method='TELEA', inpaint_radius=3, show_results=True):
      '''Denoise single image based on the non-local means algorithm.

      Inputs
      ------
      img_str:            Local path to image / String
      img_str_mask:       Local path to mask image / String
      method:             Method of inpainting (default: 'TELEA'):
                            'TELEA': Method developed by Telea in 2004 (An Image Inpainting Technique Based on the Fast Marching Method)
                            'NS': Method developed by Bertalmio et al. in 2001 (Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting)
      inpaint_radius:     Inpaint radius (Default: 3) / 1
      show_results:       Show results (Default: True) / True; False

      Outputs:
      --------
      img_inpainted:      Keypoints of detected corners
      '''

      img_original = cv.imread(img_str)
      img_mask = cv.imread(img_str_mask, cv.IMREAD_GRAYSCALE)
      
      if method == 'TELEA':
        _method = cv.INPAINT_TELEA
      elif method == 'NS':
        _method = cv.INPAINT_NS

      img_inpainted = cv.inpaint(img_original, img_mask, inpaint_radius, _method)

      if show_results:
        plt.subplot(131)
        cv.imshow('Original', img_original)
        plt.subplot(132)
        cv.imshow('Mask', img_mask)
        plt.subplot(133)
        cv.imshow('Inpainted', img_inpainted)
        plt.show()

      return img_inpainted


    def high_dynamic_range(self, folder_img, method='DEBEVEC', show_results=True):
      '''Apply High-Dynamic Range algorithms using images of different (but known) light exposures.

      Inputs
      ------
      folder_img:             Folder containing images to get calibration data
      method:                 Method of fusion as string (default: 'DEBEVEC'):
                              'DEBEVEC': Debevec method
                              'ROBERTSON': Robertson method
                              'MERTENS': Mertens method
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      image_hdr:              HDR image
      '''  

      # Loading exposure images into a list
      #img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
      images = os.listdir(folder_img)
      exposure_times = [i.split('_')[3] for i in images]
      exposure_times = [i.split('.JPG')[0] for i in exposure_times]
      exposure_times = np.array(list(map(float, exposure_times)), dtype=np.float32)
      for i in range(len(images)):
        images[i] = folder_img + images[i]
      img_list = [cv.imread(fn) for fn in images]
      #exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)


      # Merge exposures to HDR image
      if method == 'DEBEVEC':
        merge = cv.createMergeDebevec()
        hdr = merge.process(img_list, times=exposure_times.copy())
        tonemap = cv.createTonemap(gamma=2.2)
        res_32bit = tonemap.process(hdr.copy())
      elif  method == 'ROBERTSON':
        merge = cv.createMergeRobertson()
        hdr = merge.process(img_list, times=exposure_times.copy())
        tonemap = cv.createTonemap(gamma=2.2)
        res_32bit = tonemap.process(hdr.copy())
      elif method == 'MERTENS':
        merge_mertens = cv.createMergeMertens()
        res_32bit = merge_mertens.process(img_list)

      res_8bit = np.clip(res_32bit*255, 0, 255).astype('uint8')

      if show_results:
        res_8bit_resized = cv.resize(res_8bit, (800, 600))
        cv.imshow('HDR image (800 x 600)', res_8bit_resized)
        plt.show()

      image_hdr = res_8bit    

      return image_hdr


    # Machine learning
    #-----------------

    def knn_classificator(self, train_data=np.random.randint(0, 100,(25,2)).astype(np.float32), 
                          label_data=np.random.randint(0, 2,(25,1)).astype(np.float32), 
                          prediction_data=np.random.randint(0, 100,(1,2)).astype(np.float32), 
                          n_knn=3,
                          show_results=True):
      '''Apply k-Nearest Neighbor (kNN) algorithm kNN to prediction data for single-class classification 
      after performing training using labeled data.

      Inputs
      ------
      folder_img:             Folder containing images to get calibration data
      train_data:             Training data (default: np.random.randint(0, 100,(25,2)).astype(np.float32)). Rows are instances, columns are features. / 1
      label_data:             Label data (default: np.random.randint(0, 2,(25,1)).astype(np.float32)). Rows are instances, only one class (column) 
                              is allowed. Classes have to be integers. / 1
      prediction_data:        Data for predicting class (default: np.random.randint(0, 100,(1,2)).astype(np.float32)) / 1
      n_knn:                  Number of nearest neigbors (default: 3) / 1
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      class_predicted:        Predicted class / 1
      '''
  
      
      # Train model and predict
      label_data = (label_data).astype(int)
      knn = cv.ml.KNearest_create()
      knn.train(train_data, cv.ml.ROW_SAMPLE, label_data)
      _, class_predicted, neighbors, distance = knn.findNearest(prediction_data, n_knn)
      
      if show_results:
        if train_data.shape[1] == 2:

          # Separate data based on classes for plotting
          n_classes = int(max(label_data.ravel()) + 1)
          train_data_by_class = []
          for i in range(n_classes):
            train_data_by_class.append(train_data[label_data.ravel()==i])
            plt.scatter(train_data_by_class[i][:,0], train_data_by_class[i][:,1], 80, list(clrs.BASE_COLORS.keys())[i], 'o')

          plt.scatter(prediction_data[:,0], prediction_data[:,1], 80, 'k', 's')
          class_predicted_text = list(clrs.BASE_COLORS.keys())[class_predicted.astype(int)[0][0]]
          plt.title(f'Class-related samples in feature space (Predicted class: {class_predicted_text})')
          plt.ylabel('Feature 1')
          plt.xlabel('Feature 2')
          plt.show()
        else:
          raise ValueError('Only 2d feature space can be plotted with this function.')

        print( "Predicted class:  {}\n".format(class_predicted))
        print( "Neighbours:  {}\n".format(neighbors))
        print( "Distance:  {}\n".format(distance))
 
      return class_predicted


    def ocr_knn_hand_written_digits(self, img_str, n_knn=5, show_results=True):
      '''Apply Optical Character Recognition (OCR) technique to hand-written digits based on kNN.

      Inputs
      ------
      img_str:                Local path to image / String
      n_knn:                  Number of nearest neigbors (default: 5) / 1
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      class_predicted:        Predicted class / 1
      '''

      img = cv.imread(img_str)
      gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
      
      # Now we split the image to 5000 cells, each 20x20 size
      cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
      
      # Make it into a Numpy array: its size will be (50,100,20,20)
      x = np.array(cells)
      
      # Now we prepare the training data and test data
      train = x[:, :50].reshape(-1, 400).astype(np.float32) # Size = (2500,400)
      test = x[:, 50:100].reshape(-1, 400).astype(np.float32) # Size = (2500,400)
      
      # Create labels for train and test data
      k = np.arange(10)
      train_labels = np.repeat(k, 250)[:,np.newaxis]
      test_labels = train_labels.copy()
      
      # Initiate kNN, train it on the training data, then test it with the test data with k=1
      knn = cv.ml.KNearest_create()
      knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
      ret, result, neighbours, dist = knn.findNearest(test, k=n_knn)
      i_random = random.randint(0, test.shape[0])
      _, class_predicted, _, _ = knn.findNearest(test[i_random, :][np.newaxis, :], k=n_knn)
      
      # Now we check the accuracy of classification
      # For that, compare the result with test_labels and check which are wrong
      matches = result==test_labels
      correct = np.count_nonzero(matches)
      accuracy = correct*100.0/result.size
      if show_results:
        print(f"Accuracy: {accuracy}")
        print(f"Random sample (Test index: {i_random}) / Ground truth: {test_labels[i_random]} / Predicted: {class_predicted.astype(int)}")

      return class_predicted


    def ocr_svm_hand_written_digits(self, img_str, show_results=True):
      '''Apply Optical Character Recognition (OCR) technique to hand-written digits based on SVM.

      Inputs
      ------
      img_str:                Local path to image / String
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      class_predicted:        Predicted class / 1
      '''

      SZ = 20
      bin_n = 16 # Number of bins
      affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
      
      def deskew(img):
          m = cv.moments(img)
          if abs(m['mu02']) < 1e-2:
              return img.copy()
          skew = m['mu11']/m['mu02']
          M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
          img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
          return img
      
      def hog(img):
          gx = cv.Sobel(img, cv.CV_32F, 1, 0)
          gy = cv.Sobel(img, cv.CV_32F, 0, 1)
          mag, ang = cv.cartToPolar(gx, gy)
          bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
          bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
          mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
          hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
          hist = np.hstack(hists)     # hist is a 64 bit vector
          return hist
      
      img = cv.imread(cv.samples.findFile(img_str),0)
      if img is None:
          raise Exception("we need the digits.png image from samples/data here !")
      
      cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
      
      # First half is trainData, remaining is testData
      train_cells = [ i[:50] for i in cells ]
      test_cells = [ i[50:] for i in cells]
      
      deskewed = [list(map(deskew,row)) for row in train_cells]
      hogdata = [list(map(hog,row)) for row in deskewed]
      trainData = np.float32(hogdata).reshape(-1,64)
      train_labels = np.repeat(np.arange(10), 250)[:,np.newaxis]
      test_labels = train_labels.copy()
      
      svm = cv.ml.SVM_create()
      svm.setKernel(cv.ml.SVM_LINEAR)
      svm.setType(cv.ml.SVM_C_SVC)
      svm.setC(2.67)
      svm.setGamma(5.383)
      
      svm.train(trainData, cv.ml.ROW_SAMPLE, train_labels)
      svm.save('svm_data.dat')
      
      deskewed = [list(map(deskew,row)) for row in test_cells]
      hogdata = [list(map(hog,row)) for row in deskewed]
      testData = np.float32(hogdata).reshape(-1,bin_n*4)
      result = svm.predict(testData)[1]

      i_random = random.randint(0, testData.shape[0])
      result = svm.predict(testData)[1]
      class_predicted = svm.predict(testData[i_random, :][np.newaxis, :])[1]
      
      mask = result==train_labels
      correct = np.count_nonzero(mask)
      accuracy = correct*100.0/result.size

      if show_results:
        print(f"Accuracy: {accuracy}")
        print(f"Random sample (Test index: {i_random}) / Ground truth: {test_labels[i_random]} / Predicted: {class_predicted.astype(int)}")

      return class_predicted


    def kmeans_clustering(self, train_data=np.hstack(([np.random.randint(25, 100, 25), np.random.randint(175, 255, 25)])), 
                            type_criteria='EPS ITER',
                            max_iter=10,
                            eps=1.0,
                            attempts=10,
                            n_clusters=2,
                            show_results=True):
      '''Apply K-Means clustering to the given data. This function requires up to 2 features for plotting the results.

      Inputs
      ------
      folder_img:             Folder containing images to get calibration data
      train_data:             Training data (default: np.random.randint(0, 100,(25,2)).astype(np.float32)). Rows are instances, columns are features. / 1
      type_criteria:          Type of criteria (default: 'EPS ITER'):
                              'EPS': Stop izeration only when specified accuracy is reached
                              'ITER': Stop izeration if specified number of iterations is reached
                              'EPS ITER': Stop iteration if any of the above conditions is met
      max_iter:               Maximum number of iterations (default: 10) / 1
      eps:                    Accuracy tolerance (default: 1.0) / 1.0
      attempts:               Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns 
                              the labels that yield the best compactness (default: 10) / True; False
      n_clusters:             Number of clusters (default: 2) / 1
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      labels:                 Cluster labels / 1
      centers:                Cluster centers / 1
      '''

      if type_criteria == 'EPS':
        _criteria = cv.TERM_CRITERIA_EPS
      elif type_criteria == 'ITER':
        _criteria = cv.TERM_CRITERIA_MAX_ITER
      elif type_criteria == 'EPS ITER':
        _criteria = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
      else:
        raise TypeError('Type of criterion unknown.')
      criteria = (_criteria, max_iter, eps)
      flags = cv.KMEANS_RANDOM_CENTERS

      if len(train_data.shape)==1:
        n_features = 1
      else:
        n_features = train_data.shape[1]

      if n_features == 1:
        z = train_data.reshape((50, 1))
        z = np.float32(z)
        compactness, labels, centers = cv.kmeans(z, n_clusters, None, criteria, attempts, flags)

        if show_results:
          train_data_by_class = []
          for i in range(n_clusters):
            train_data_by_class.append(z[labels==i])
            plt.hist(train_data_by_class[i], 256, [0, 256], color=list(clrs.BASE_COLORS.keys())[i])
          plt.hist(centers, 32, [0, 256], color='k')
          plt.title('Clustering')
          plt.show()

      elif n_features == 2:

        Z = np.float32(train_data)
        ret, labels, centers = cv.kmeans(Z, n_clusters, None, criteria, attempts, flags)
        
        if show_results:
          train_data_by_class = []
          for i in range(n_clusters):
            train_data_by_class.append(Z[labels.ravel()==i])
            plt.scatter(train_data_by_class[i][:,0], train_data_by_class[i][:,1], 80, list(clrs.BASE_COLORS.keys())[i], 'o')

          plt.scatter(centers[:, 0], centers[:, 1], s=80, c='y', marker='s')
          plt.xlabel('Feature 1')
          plt.ylabel('Feature 2')
          plt.title('Clustering')
          plt.show()

      else:
        raise TypeError('Data with higher dimensions can not be processed in this function.')
      
      return labels, centers


    def color_quantization(self, img_str, 
                           type_criteria='EPS ITER',
                           max_iter=10,
                           eps=1.0,
                           attempts=10,
                           n_clusters=8,
                           show_results=True):
      '''Apply color quantization of image.

      Inputs
      ------
      img_str:                Local path to image / String
      type_criteria:          Type of criteria (default: 'EPS ITER'):
                              'EPS': Stop izeration only when specified accuracy is reached
                              'ITER': Stop izeration if specified number of iterations is reached
                              'EPS ITER': Stop iteration if any of the above conditions is met
      max_iter:               Maximum number of iterations (default: 10) / 1
      eps:                    Accuracy tolerance (default: 1.0) / 1.0
      attempts:               Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns 
                              the labels that yield the best compactness  (default: 10) / True; False
      n_clusters:             Number of clusters (default: 8) / 1
      show_results:           Show results (Default: True) / True; False

      Outputs:
      --------
      image_quantized:        Quantized image / Image
      '''

      # Set criteria
      if type_criteria == 'EPS':
        _criteria = cv.TERM_CRITERIA_EPS
      elif type_criteria == 'ITER':
        _criteria = cv.TERM_CRITERIA_MAX_ITER
      elif type_criteria == 'EPS ITER':
        _criteria = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
      else:
        raise TypeError('Type of criterion unknown.')
      criteria = (_criteria, max_iter, eps)
      flags = cv.KMEANS_RANDOM_CENTERS

      # Get image
      img_original = cv.imread(img_str)
      Z = img_original.reshape((-1, 3))
      Z = np.float32(Z)
      
      # Define criteria, number of clusters(K) and apply kmeans()
      ret,label,center=cv.kmeans(Z, n_clusters, None, criteria, attempts, flags)
      
      # Convert back into uint8, and make original image
      center = np.uint8(center)
      res = center[label.flatten()]
      image_quantized = res.reshape((img_original.shape))
      
      if show_results:
        plt.subplot(121)
        cv.imshow('Original image', img_original)
        plt.subplot(122)
        cv.imshow('Image quantized', image_quantized)

      return image_quantized

    # Deep learning
    #-----------------

    def create_images_labels_yolo_from_dataset(self, data, split):
      '''Create images and labels in YOLO data format from dataset'''

      data = data[split]
      for i, example in enumerate(data):
        image = example["image"]
        labels = example["litter"]["label"]
        bboxes = example["litter"]["bbox"]
        targets = []
        for label, box in zip(labels, bboxes):
          targets.append(f"{label} {box[0]} {box[1]} {box[2]} {box[3]}")
          with open(f"datasets/labels/{split}/{i}.txt", "w") as f:
            for target in targets:
              f.write(target + "\n")
            image.save(f"datasets/images/{split}/")


    def object_detection_yolo_v8(self, model="yolov8m.pt", retrain=True, path_predict_image=[], path_yaml_file=[], n_epochs_train=30, optimizer="Adam", batch=4, lr0=1e-3):
      '''Use pretrained model for object detection YOLO v8 from Ultralytics and apply a prediction. Retrain model if desired.
      
      Inputs
      ------
      model:                YOLO v8 model (default: "yolov8m.pt")
      retrain:              Retrain (default: True): Load pretrained model and retrain using dataset from a specific folder or dataset from open datasets.
      path_predict_image:   Path to file for prediction (default: [])
      path_yaml_file:       Path to YAML file (default: [])
      path_folder_training: Path to folder containing training images (default: []) using YAML file or to sources on the network.
      name_open_dataset:    Name of dataset using open dataset function
      n_epochs_train:       Number of epochs for training (default: 30)
      optimizer:            Optimizer (default: "Adam")
      batch:                Batch number (default: 4)
      lr0:                  Regularization parameter (default: 1e-3)
      
      Outputs
      -------
      '''

      # Get pretrained medium-sized model for object detection (80 classes)
      model = YOLO(model)

      # Retrain model with using data folder if desired
      if retrain:
        if path_yaml_file != []:
          results_training = model.train(data=path_yaml_file, epochs=n_epochs_train, optimizer=optimizer, batch=batch, lr0=lr0)
        else:
          results_training = model.train(data=path_yaml_file, epochs=n_epochs_train, optimizer=optimizer, batch=batch, lr0=lr0)

      # Evaluate model performance on validation set
      metrics = model.val()

      # Predict
      results_prediction = model.predict(path_predict_image)
      result_prediction = results_prediction[0]

      # Print results
      print('Detected objects:')
      if len(result_prediction.boxes.cls) > 0:
        for i in range(len(result_prediction.boxes.cls)):
          print(f'Class: {result_prediction.names[int(result_prediction.boxes.cls[i])]}')
          print(f'Confidence: {result_prediction.boxes.conf[i]}')
          print(f'Box coordinates: {result_prediction.boxes.xyxy[i]}')
      else:
        print('No objects detected.')

      # Export model to ONNX format
      path = model.export(format="onnx")


    def classification_dl(self, data_dir, sz_image=200, color_mode='grayscale', label_mode='categorical', fraction_train=0.7, fraction_validation=0.2, fraction_test=0.1, dl_method='cnn', dl_parms_dict={'type_cnn': 1, 'activation_cnn_layer': 'relu', 'activation_cnn_out': 'softmax'}, optimizer='adam', n_batch=32, n_epoch=100, learning_rate=1e-5, n_patience=5, pretrain_or_load=False, early_stopping=False, save_model=True, export_training_performance_image=True, export_confusion_matrix_image=True):

      # Define constants
      AVERAGING_METRICS   = ['micro', 'macro', 'weighted']   # Sample weights for performance measures
      WEIGHTS_COHEN_KAPPA = ['linear', 'quadratic']          # Weights of Cohen Kappa metric

      # Get date and time
      date_time = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
      print("Date/Time: " + date_time)

      # Apply HW and SW checks
      print(f'HW / SW CHECKS')
      print("---------------------------------------")
      # Show tensorflow version
      print("---------------------------------------")
      print(f'Tensorflow version: {tf.__version__}')
      print("---------------------------------------")
      print(f'GPU detection:')
      print("---------------------------------------")
      # Avoid OOM errors by setting GPU Memory Consumption Growth
      gpus = tf.config.list_physical_devices('GPU')
      lpd = tf.config.list_physical_devices()
      lld = tf.config.list_logical_devices()
      if gpus:
          for gpu in gpus: 
              tf.config.experimental.set_memory_growth(gpu, True)
          print("At least one GPU found.")
      else:
          print("No GPU detected.")
      print("---------------------------------------")

      # Import and check  images
      print(f'IMAGE IMPORT AND CHECK')
      print("---------------------------------------")
      # Remove dodgy images
      print("Checking images...")
      n_images_all, class_name_list = check_images(data_dir)
      n_class = len(class_name_list)
      print("Image checks completed.")

      print("Importing images from " + data_dir + "...")
      data = keras.utils.image_dataset_from_directory(directory=data_dir, 
                                                      label_mode=label_mode,
                                                      color_mode=color_mode,
                                                      batch_size=n_images_all,
                                                      image_size=(sz_image, sz_image))
      print("Image import completed.")
      print("---------------------------------------")

      # Apply image preprocessing
      print(f'PREPROCESSING')
      print("---------------------------------------")
      # Convert data to numpy arrays
      print("Converting images to numpy arrays...")
      x_np, y_np = convert_batchdataset_to_numpy(data)
      print("Conversion completed.")
      print("---------------------------------------")

      # Split data to get datasets for training, validation and test
      print(f'SPLITTING')
      print("---------------------------------------")
      print("Splitting datasets for training, validation and test...")
      (x_train, x_test_aux, 
      y_train, y_test_aux) = sklearn.model_selection.train_test_split(x_np, y_np, 
                                                                      train_size=fraction_train)
      (x_val, x_test, 
      y_val, y_test) = sklearn.model_selection.train_test_split(x_test_aux, y_test_aux, 
                                                                test_size=fraction_test/((fraction_test + fraction_validation)))
      print("---------------------------------------")

      # Get sparse labels of training, validation and test datasets
      print("Transforming labels into sparse...")
      y_train_sparse = np.argmax(y_train, axis=1)
      y_val_sparse = np.argmax(y_val, axis=1)
      y_test_sparse = np.argmax(y_test, axis=1)
      print("Transformation completed...")

      # Get number of iterations (steps) per batch for reporting
      n_iterations = n_images_all/n_batch
      print("Number of iterations (" + str(n_images_all) + "/" + str(n_batch) + ") = " + str(n_iterations))

      # Clear session
      keras.backend.clear_session()

      # Set up sequential model and compile
      print(f'MODEL')
      print("---------------------------------------")
      print("Set up model...")
      if dl_method == 'cnn':
          
        # Build CNN model
        print("Building CNN model...")
        model = build_cnn_model(dl_parms_dict['type_cnn'], dl_parms_dict['activation_cnn_layer'], dl_parms_dict['activation_cnn_out'], sz_image, color_mode, n_class, optimizer, learning_rate)

        # Show model information
        model.summary()

      elif dl_method == 1:
          
        # Define sequential RNN model of specific type
        print("Building RNN model...")
        model = build_rnn_model(dl_parms_dict['type_rnn'], sz_image, n_class, optimizer, learning_rate)
        
        # Show model information
        model.summary()

      elif dl_method == 2:
          
        # Define sequential AE model of specific type
        print("Building AE model...")
        model_cae, model = build_ae_model(dl_parms_dict['type_ae'], dl_parms_dict['type_classifier'], color_mode, sz_image, optimizer, learning_rate, dl_parms_dict['latent_dim'], dl_parms_dict['n_units_clf'], n_class)

        # Show model information
        model_cae.summary()
        model.summary()

      elif dl_method == 3:
          
        # Define GAN model
        print("Building conditional GAN model...")
        if color_mode == 'grayscale':
          sz_image_total = (sz_image, sz_image, 1)
        elif color_mode == 'rgb':
          sz_image_total = (sz_image, sz_image, 3)
        generator, discriminator, cgan = build_cgan_model(sz_image_total, learning_rate, dl_parms_dict['latent_dim'], n_class)

        # Show model information
        generator.summary()
        discriminator.summary()
        cgan.summary()

        # Show model graphics
        # plot_model(generator, show_shapes=True, show_layer_names=True)
        # plot_model(discriminator, show_shapes=True, show_layer_names=True)
        # plot_model(cgan, show_shapes=True, show_layer_names=True)

      print("Model setup competed.")
      print("---------------------------------------")

      print(f'TRAINING:')
      print("---------------------------------------")

      # Start timer
      time_start = time.time()

      # Set callbacks
      callbacks = tf.keras.callbacks.TensorBoard(log_dir='logs')
      if early_stopping:
          early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=n_patience)
          callbacks = [callbacks, early_stopping_callback]

      # Fit model to training data
      if dl_method == 'cnn' or dl_method == 'rnn':
          
          # Train CNN or RNN variant model
          print("Training classifier...")
          hist = model.fit(x_train,
                          y_train,
                          batch_size=n_batch,
                          epochs=n_epoch, 
                          validation_data=(x_val, y_val),
                          callbacks=callbacks)
          
      elif dl_method == 'ae':
          
          # Train autoencoder variant model
          if pretrain_or_load:
            print("Pretraining Autoencoder...")
            hist_ae = model_cae.fit(x_train,
                                    x_train,
                                    batch_size=n_batch,
                                    epochs=n_epoch, 
                                    validation_data=(x_val, x_val),
                                    callbacks=callbacks)
            model_cae.save_weights('model_cae.h5')
          else:
            model_cae.load_weights('model_cae.h5')

          # Copy weights
          n_layers_encoder = 13
          for l1,l2 in zip(model.layers[0:n_layers_encoder], model_cae.layers[0:n_layers_encoder]):
              l1.set_weights(l2.get_weights())

          # Plot weights of encoder and classifier (encoder + dense-network) to check if copied correctly
          print(f'Encoder / Weights of first layer: {model_cae.get_weights()[0]}')
          print(f'Encoder / Biases of first layer: {model_cae.get_weights()[1]}')
          print(f'Model (Classifier) / Weights of first layer: {model.get_weights()[0]}')
          print(f'Model (Classifier) / Biases of first layer: {model.get_weights()[1]}')

          # Set weights of encoder-specific layers of classifier to be untrainable
          for layer in model.layers[0:n_layers_encoder]:
              layer.trainable = False

          # Compile classifier model
          model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                        metrics=['accuracy'],
                        loss='categorical_crossentropy')

          # Train classifier model
          print("Training classifier...")
          hist = model.fit(x_train,
                           y_train,
                           batch_size=n_batch,
                           epochs=n_epoch,
                           validation_data=(x_val, y_val),
                           callbacks=callbacks)
          
      elif dl_method == 'gan':

        # Train GAN model
        model_discriminator, model = train_cgan_model(generator, discriminator, cgan, [x_train, y_train], 
                                                      n_class, noise_size=dl_parms_dict['latent_dim'], n_epochs=n_epoch, n_batch=n_batch)

      # End timer
      time_end = time.time()
      time_training = time_end - time_start

      # Save model if desired
      print(f'SAVE')
      print("---------------------------------------")
      if save_model:
        print("Saving model...")
        if dl_method == 'cnn' or dl_method == 'rnn':
            model.save_weights(os.path.join('models','model.h5'))
        elif dl_method == 'ae':
            model_cae.save_weights(os.path.join('models','model_cae.h5'))
            model.save_weights(os.path.join('models','model.h5'))

      # Evaluate trained model performance
      print(f'EVALUATE')
      print("---------------------------------------")
      # Show performance: loss function
      name_computation = dl_method
      if dl_method == 'cnn' or dl_method == 'rnn' or dl_method == 'ae' or dl_method == 'gan':
        plot_history(hist, 
                     name_computation,
                     export_figure=export_training_performance_image,
                     date_time=date_time)

      # Evaluate based on test dataset
      results = model.evaluate(x_test, 
                              y_test, 
                              batch_size=n_batch, 
                              verbose=2)

      # Predict for all test instances
      y_test_pred = model.predict(x_test)
      y_test_pred_sparse = np.argmax(y_test_pred, axis=1)

      # Get confusion matrix based on test dataset and plot
      confusion_mat_test = plot_confusion_mat(hist, 
                                              name_computation,
                                              y_test_sparse, 
                                              y_test_pred_sparse,
                                              export_figure=export_confusion_matrix_image,
                                              date_time=date_time)

      # Get metrics based on test dataset
      precision_test   = [None] * len(AVERAGING_METRICS)
      recall_test      = [None] * len(AVERAGING_METRICS)
      f1_test          = [None] * len(AVERAGING_METRICS)
      cohen_kappa_test = [None] * len(WEIGHTS_COHEN_KAPPA)

      accuracy_test = sklearn.metrics.accuracy_score(y_test_sparse, y_test_pred_sparse)
      mcc_test = sklearn.metrics.matthews_corrcoef(y_test_sparse, y_test_pred_sparse)

      for i in range(len(AVERAGING_METRICS)):
        precision_test[i] = sklearn.metrics.precision_score(y_test_sparse, 
                                                            y_test_pred_sparse, 
                                                            average=AVERAGING_METRICS[i])
        recall_test[i] = sklearn.metrics.recall_score(y_test_sparse, 
                                                      y_test_pred_sparse, 
                                                      average=AVERAGING_METRICS[i])
        f1_test[i] = sklearn.metrics.f1_score(y_test_sparse, 
                                              y_test_pred_sparse, 
                                              average=AVERAGING_METRICS[i])
        
      # fpr_test, tpr_test, _ = roc_curve(y_test_sparse, 
      #                                   y_test_pred_sparse)

      # auc_test = auc(fpr_test,tpr_test)

      for i in range(len(WEIGHTS_COHEN_KAPPA)):
        cohen_kappa_test[i] = sklearn.metrics.cohen_kappa_score(y_test_sparse, 
                                                                y_test_pred_sparse, 
                                                                weights=WEIGHTS_COHEN_KAPPA[i])

      # Get number of total parameters, trainable and non-trainable parameters
      #num_parms_trainable = K.count_params(model.trainable_weights)
      #num_parms_non_trainable = K.count_params(model.non_trainable_weights)
      num_trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
      num_total_params = sum([tf.size(w).numpy() for w in model.weights])
      num_non_trainable_params = num_total_params - num_trainable_params

      print('------------')
      print('Test results')
      print('------------')
      print('Accuracy = ' + str(accuracy_test))
      print('Precision = ' +  str(precision_test))
      print('Recall = ' +  str(recall_test))
      print('F1 = ' +  str(f1_test))
      print('MCC = ' +  str(mcc_test))
      print('Cohen Kappa = ' +  str(cohen_kappa_test))

      # Predict for a random image of the test data set and display results
      print('-----------------------------')
      print('Random single test prediction')
      print('-----------------------------')
      y_test_pred_single = model.predict(x_test[:1])
      y_test_pred_single_sparse = np.argmax(y_test_pred_single, axis=1)

      print('Most probable activity is ' + "\'" + class_name_list[y_test_pred_single_sparse[0].astype(int)] + "'.")
          
      print('------------------')
      print('Additional metrics')
      print('------------------')
      print('Elapsed execution time for training: ' + str(time_training) + ' s')

      print('-------------')
      print('Miscellaneous')
      print('-------------')
      print('Number of total parameters: ' + str(num_total_params))
      print('Number of trainable parameters: ' + str(num_trainable_params))
      print('Number of non-trainable parameters: ' + str(num_non_trainable_params))

      print("Done.")


    def basler_camera_configuration(self, parms):
      '''Configurate BASLER camera using the Pypylon wrapper framework which uses the GenICam standard. 
      Images are grabbed consecutively and saved to a list. See 
      'https://www.baslerweb.com/de-de/learning/pypylon/vod/' for more information.
      
      Inputs
      ------
      parms:                  Parameter object for BASLER camera configuration:
        - idx_device:         Index of device to select, if more than one active device was found (default: 0).

      Outputs
      -------
      camera:                 Camera object
      devices:                List of devices
      '''
      
      print("----------------------------")
      print(f"BASLER camera configuration")
      print("----------------------------")

      # Extract parameters
      idx_device = parms['idx_device']

      # Get camera object (using transport layers) if on
      devices = pylon.TlFactory.GetInstance().EnumerateDevices()
      if len(devices) == 1:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        print(f"One device found:")
        print(f"Model name: {devices[0].GetModelName()}; SN: {devices[0].GetSerialNumber()}")
      elif len(devices) > 1:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[idx_device]))
        print(f"Number of devices found: {len(devices)}")
        for d in devices:
          print(f"Model name: {d.GetModelName()}; SN: {d.GetSerialNumber()}")
          print(f"Selected device: {devices[idx_device].GetModelName()}; SN: {devices[idx_device].GetSerialNumber()}")
      else:
        print(f"No devices found.")

      return camera, devices


    def basler_camera_connect(self, camera):
      '''Connect to Basler camera.
      
      Inputs
      ------
      camera:   Basler camera object

      Outputs
      -------
      camera:   Basler camera object
      '''

      camera.Open()

      return camera
    

    def basler_camera_disconnect(self, camera):
      '''Disconnect from Basler camera.
      
      Inputs
      ------
      source:   Terasense source object

      Outputs
      -------
      camera:   Basler camera object
      '''

      camera.Close()

      return camera

    # Hardware-based applications
    #----------------------------
    def grab_software_triggered_camera_basler(self, parms, camera, devices, show_image=False):
      '''This function implements a simple application for BASLER cameras using the Pypylon wrapper 
      framework which uses the GenICam standard. Images are grabbed consecutively and saved to 
      a list. See 'https://www.baslerweb.com/de-de/learning/pypylon/vod/' for more information.
      
      Inputs
      ------
      parms:                  Parameter object for BASLER camera configuration:
        - mode:               Type of operation (default: 'foreground_loop'):
                                'foreground_loop': Operations take place in the foreground, i.e. Python
                                grabs all images and blocks following operations until the acquisition is 
                                finished.
                                'background_loop': Operations of images grabbing are applied in the 
                                background
        - n_images:           Number of images to grab (default: 10)
        - timeout_ms:         Time out in milliseconds (default: 5000)
        - show_stats:         Show statistics of images grabbed (default: True)
        - show_one_image:     Flag to show one image (default: True)
        - idx_show_one_image: Index for the image to show (default: 0)
        - idx_device:         Index of device to select, if more than one active device was found (default: 0).
      camera:                 Basler camera object
      devices:                List of devices
      '''
      
      print("----------------------------")
      print(f"BASLER camera image grabber")
      print("----------------------------")

      # Extract parameters
      mode = parms['mode']
      n_images = parms['n_images']
      timeout_ms = parms['timeout_ms']
      show_one_image = parms['show_one_image']
      idx_show_one_image = parms['idx_show_one_image']

      # Configuration
      #--------------
      # Get some features of the current devices
      # print(f"Gain: {camera.Gain.Value}")
      # print(f"Trigger selector: {camera.TriggerSelector.Value}")
      # print(f"(Symbolics of trigger selector: {camera.TriggerSelector.Symbolics})")
      # print(f"Pixel format: {camera.PixelFormat.Value}")
      # print(f"Exposure time: {camera.ExposureTime.Value}")

      # Demonstrate some feature access
      # new_width = camera.Width.Value - camera.Width.Inc
      # if new_width >= camera.Width.Min:
      #     camera.Width.Value = new_width

      # Operation
      #----------
      # Apply operations int the foreground
      images = []
      if mode == 'foreground_loop':
        
        camera.StartGrabbingMax(n_images)
        print(f"MV: Collecting {n_images} images:")

        while camera.IsGrabbing():
            
            # Grab images
            grabResult = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)

            # Show statistics
            try:
              if grabResult.GrabSucceeded():
                  images.append(cv.cvtColor(grabResult.Array, cv.COLOR_BGR2RGB))
              else:
                raise RuntimeError("image not grabbed.")
            except Exception as e:
              raise RuntimeError(e)

            # Release current grab
            grabResult.Release()

      elif mode == 'background_loop':

        # Define image event handler class
        class ImageHandler(pylon.ImageEventHandler):
          def __init__(self):
            super().__init__()
            self.img_sum = np.zeros((camera.Height.Value, camera.Width.Value), dtype=np.uint16)

          def OnImageGrabbed(self, camera, grabResult):
            try:
              if grabResult.GrabSucceeded():
                #images.append(grabResult.Array)
                images.append(cv.cvtColor(grabResult.Array, cv.COLOR_BGR2RGB))
              else:
                raise RuntimeError("Grab failed.")
            except Exception as e:
              print(e)
              #traceback.print_exc()

        # Instantiate callback handler
        handler = ImageHandler()

        # Register with the pylon loop
        camera.RegisterImageEventHandler(handler, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)

        # Fetch some images in the background
        camera.StartGrabbingMax(n_images, pylon.GrabStrategy_LatestImages, pylon.GrabLoop_ProvidedByInstantCamera)
        
        # Do something else while the images are grabbed (line before)
        while camera.IsGrabbing():
          time.sleep(0.1)

        camera.StopGrabbing()
        camera.DeregisterCameraEventHandler(handler)
      
      print(f"MV: Images collected.")

      # Show one of the images based on the selected index if desired
      if show_one_image:
        plt.imshow(images[idx_show_one_image])
        plt.axis('off')
        plt.title(f"Image {idx_show_one_image} from BASLER Camera {devices[0].GetModelName()}(SN: {devices[0].GetSerialNumber()})")
        plt.show(block=False)

      images = np.concatenate(images, axis=0)

      if show_image:
        plt.figure(figsize=(10,7))
        plt.imshow(images)
        plt.show(block=False)

      return images, camera

    def grab_software_triggered_camera_basler_julia(self, parms, camera, devices, show_image=False):
      '''Grab images from a Basler camera using software triggering.'''

      # Extract parameters
      mode            = parms.get('mode', 'foreground_loop')
      n_images        = parms.get('n_images', 10)
      timeout_ms      = parms.get('timeout_ms', 5000)
      show_one_image  = parms.get('show_one_image', True)
      idx_show_image  = parms.get('idx_show_one_image', 0)

      # --- 1) Configure camera for software trigger ---
      camera.TriggerSelector.SetValue('FrameStart')
      camera.TriggerSource.SetValue('Software')
      camera.TriggerMode.SetValue('On')
      camera.AcquisitionMode.SetValue('Continuous')
      # (optionally set exposures/gains here)

      images = []

      if mode == 'foreground_loop':
          camera.StartGrabbingMax(n_images)

          for i in range(n_images):
              # Fire the trigger
              camera.ExecuteSoftwareTrigger()

              # Wait for the frame
              grabResult = camera.RetrieveResult(timeout_ms,
                                                pylon.TimeoutHandling_ThrowException)
              if not grabResult.GrabSucceeded():
                  grabResult.Release()
                  raise RuntimeError(f"Grab {i} failed: "
                                    f"{grabResult.ErrorCode} {grabResult.ErrorDescription}")

              # Convert and store
              img = grabResult.Array
              images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
              grabResult.Release()

          camera.StopGrabbing()

      elif mode == 'background_loop':
          # (Register handler as you already have it, but inside OnImageGrabbed
          # you don’t need to trigger—the HW will still need the software trigger!)
          class Handler(pylon.ImageEventHandler):
              def __init__(self):
                  super().__init__()
              def OnImageGrabbed(self, cam, grabResult):
                  if grabResult.GrabSucceeded():
                      images.append(cv.cvtColor(grabResult.Array, cv.COLOR_BGR2RGB))
                  grabResult.Release()

          handler = Handler()
          camera.RegisterImageEventHandler(handler,
                                          pylon.RegistrationMode_ReplaceAll,
                                          pylon.Cleanup_None)

          camera.StartGrabbingMax(n_images,
                                  pylon.GrabStrategy_OneByOne,
                                  pylon.GrabLoop_ProvidedByInstantCamera)

          for i in range(n_images):
              camera.ExecuteSoftwareTrigger()
              time.sleep(0.01)   # give the event handler time to run

          # wait until done
          while camera.IsGrabbing():
              time.sleep(0.01)

          camera.StopGrabbing()
          camera.DeregisterCameraEventHandler(handler)

      else:
          raise ValueError(f"Unknown mode '{mode}'")

      # --- 4) Show one image if requested ---
      if show_one_image and images:
          plt.imshow(images[idx_show_image])
          plt.axis('off')
          title = (f"Image {idx_show_image} from {devices[0].GetModelName()} "
                  f"(SN: {devices[0].GetSerialNumber()})")
          plt.title(title)
          plt.show(block=False)

      # Concatenate into one big array if you really want that:
      # images = np.stack(images, axis=0)  

      return images, camera


def test_image_processing_methods(mv_1):
  '''Test image processing methods, e.g. basic operations, arithmetic operations, geometric operation, color space changes, thresholding, smoothing, morphological operations, image gradients, image pyramids, contours, histograms, image transforms, template matching, lines and circles detection, segmentation etc.
  '''

  # Basic operation methods
  # -----------------------
  img = mv_1.load_image('images\\messi5.jpg')
  cols = mv_1.read_pixel_slow('images\\messi5.jpg', 100, 100)
  img_changed = mv_1.write_pixel_slow('images\\messi5.jpg', 100, 100, [255, 255, 255])
  cols = mv_1.read_pixel_fast('images\\messi5.jpg', 99, 99)
  img_changed = mv_1.write_pixel_fast('images\\messi5.jpg', 99, 99, [255, 255, 255])
  shape, size, dtype = mv_1.get_image_properties('images\\messi5.jpg')
  roi = mv_1.extract_roi('images\\messi5.jpg', [330, 390], [280, 340])
  img_changed = mv_1.place_roi('images\\messi5.jpg', roi, [100, 160], [273, 333])
  channels = mv_1.split_channels('images\\messi5.jpg')
  img_merged = mv_1.merge_channels(channels)
  img_framed = mv_1.make_border('images\\messi5.jpg', [10, 10, 10, 10], cv.BORDER_CONSTANT, values=[255, 255, 255])

  # Arithmetic operation methods
  # ----------------------------
  img_1 = mv_1.load_image('images\\messi5.jpg')
  img_2 = mv_1.load_image('images\\messi6.jpg')
  img_add = mv_1.add_images('images\\messi5.jpg', 'images\\messi6.jpg')
  img_blend = mv_1.blend_images('images\\messi5.jpg', 'images\\messi6.jpg', 0.3, 0.7, 0.0)
  img_overlay = mv_1.overlay_images_using_mask('images\\messi5.jpg', 'opencv-logo-white.png')

  # Geometric transformations
  # -------------------------
  img_scaled = mv_1.scale(img_str='images\\messi5.jpg', coeff_scale_x=2, coeff_scale_y=2, show_results=True)
  img_translated = mv_1.translate(img_str='images\\messi5.jpg', delta_x=100, delta_y=50, show_results=True)
  img_rotated = mv_1.rotate(img_str='images\\messi5.jpg', center=None, angle=90, scale=1, show_results=True)
  img_transformed_affine = mv_1.transform_affine(img_str='images\\messi5.jpg', pts_in=np.float32([[50, 50], [200, 50], [50, 200]]), pts_out=np.float32([[10, 100], [200, 50], [100, 250]]), show_results=True)
  img_transformed_perspective= mv_1.transform_perspective(img_str='images\\messi5.jpg', pts_in=np.float32([[56, 65], [368, 52], [28, 387], [389, 390]]), pts_out=np.float32([[0, 0], [300, 0], [0, 300], [300, 300]]), size=(300, 300), show_results=True)

  # Color space change
  # ------------------
  img_color_space_converted = mv_1.convert_color_space(img_str='images\\messi5.jpg', type='RGB 2 HSV')
  img_extract = mv_1.extract_image_using_color_range('images\\bottles.jpg', [130,255,255], [110,50,50], show_results=True)

  # Thresholding
  # ------------
  image_thresholded = mv_1.threshold_simple(img_str='images\\messi5.jpg', threshold_val=127, substitute_val=255, type='BINARY', show_results=True)
  image_thresholded = mv_1.threshold_adaptive(img_str='images\\messi5.jpg', threshold_val=255, adaptive_method='ADAPTIVE THRESH MEAN', type='BINARY', size_neigborhood=11, const_subtracted=2, show_results=True)
  image_thresholded = mv_1.threshold_otsu(img_str='images\\messi5.jpg', threshold_val=127, substitute_val=255, show_results=True)

  # Smoothing
  # ---------
  image_filtered = mv_1.image_filtering_2d_convolution(img_str='images\\opencv-logo-white.png', kernel=np.ones((5, 5), np.float32)/25, show_results=True)
  image_filtered = mv_1.image_smoothing(img_str='images\\opencv-logo-white.png', mode='AVERAGE', parameter_list=[], show_results=True)

  # Morphological operations
  # ------------------------
  image_processed = mv_1.morphological_operation(img_str='images\\j.jpg', mode='EROSION', kernel=np.ones((5,5),np.uint8), kernel_type='PASSED', kernel_size=(5, 5), show_results=True)

  # Image gradients
  # ---------------
  image_gradient = mv_1.image_gradient(img_str='images\\dave.jpg', kernel_type='LAPLACIAN', kernel_size=5, show_results=True)

  # Image pyramids
  # --------------
  image_pyramids = mv_1.image_pyramids(img_str='images\\messi5.jpg', level=5, direction='DOWN', show_results=True)
  image_blended = mv_1.image_blend(img_str_1='images\\apple.jpg', img_str_2='images\\orange.jpg', alpha=0.5, beta=0.5, gamma=0, show_results=True)

  # Contours
  # --------
  contours, hierarchy = mv_1.find_contours(img_str='images\\whitebox.jpg', retrieval_mode='RETRIEVAL TREE', contour_approximation='CHAINING APPROXIMATION NONE', \
                                                threshold_val=127, substitute_val=255, show_results=True)

  (moment, area, perimeter, hull, 
  convexity, bound_rect_unrotated, 
  bound_rect_rotated, enclosing_circle, 
  enclosing_ellipse, line,
  aspect_ratio, rect_area, extent,
  solidity, equivalent_diameter,
  pixelpoints, min_max_val_loc, 
  mean_val, extreme_points, defects) = mv_1.get_contours_features_properties(img_str='images\\flash.jpg', retrieval_mode='RETRIEVAL TREE', \
                                                                                  contour_approximation='CHAINING APPROXIMATION NONE', \
                                                                                  threshold_val=127, substitute_val=255, show_results=True)

  distance_shortest = mv_1.shortest_distance_point_contour(img_str='images\\flash.jpg', point=(50, 50), retrieval_mode='RETRIEVAL TREE', \
                                                                contour_approximation='CHAINING APPROXIMATION NONE', \
                                                                threshold_val=127, substitute_val=255, show_results=True)

  similarity = mv_1.match_shapes(img_str_1='images\\star1.jpg', img_str_2='images\\star2.jpg')

  # Histograms
  # ----------
  histograms = mv_1.get_histograms_1d(img_str='images\\home.jpg', mask=None, show_results=True)
  histograms = mv_1.get_histograms_1d(img_str='images\\red_ball.jpg', mask=None, show_results=True)
  histograms = mv_1.get_histograms_1d(img_str='images\\green_ball.jpg', mask=None, show_results=True)
  histograms = mv_1.get_histograms_1d(img_str='images\\blue_ball.jpg', mask=None, show_results=True)
  image_equalized = mv_1.equalize_histogram(img_str='images\\tsukuba_l.png', mode='WHOLE', clip_limit=2.0, tile_grid_size=(8, 8), show_results=True)
  image_equalized = mv_1.equalize_histogram(img_str='images\\tsukuba_l.png', mode='TILES', clip_limit=2.0, tile_grid_size=(8, 8), show_results=True)
  histograms_2d = mv_1.get_histogram_2d(img_str='images\\home.jpg', mask=None, bins=[180, 256], range=[0, 180, 0, 255], show_results=True)
  image_backprojected = mv_1.backproject_histogram(img_str_roi='images\\grass.jpg', img_str_target='images\\messi5.jpg', mask=None, bins=[180, 256], range=[0, 180, 0, 255], show_results=True)

  # Image transforms
  # -----------------
  image_filtered = mv_1.filter_fourier(img_str='images\\messi5.jpg', type='HIGH PASS', size_lpf=30, show_results=True)
  image_filtered = mv_1.filter_fourier(img_str='images\\messi5.jpg', type='LOW PASS', size_lpf=30, show_results=True)

  # Template matching
  # -----------------
  template_locations = mv_1.template_matching(img_str='images\\mario.jpg', template_str='images\\mario_coin.jpg', method='TM_CCOEFF_NORMED', show_results=True)

  # Lines and circles detection
  # ---------------------------
  lines_hough = mv_1.find_hough_lines(img_str='images\\sudoku.jpg', canny_thd_1=50, canny_thd_2=150, sobel_size=3, rho_res=1, theta_res=np.pi/180, threshold=200, show_results=True)
  circles_hough = mv_1.find_hough_circle(img_str='images\\opencv-logo-white.png', blur_kernel_size=5, method='HOUGH GRADIENT', accumulator_resolution=1, min_distance=20, param_1=50, param_2=30, min_radius=0, max_radius=0, show_results=True)

  # Segmentation
  # ------------
  image_watershed = mv_1.segment_watershed(img_str='images\\coins.jpg', size_kernel_opening_dilation=3, n_iter_opening=2, n_iter_dilation=3, mask_size_distance_transform=5, threshold_coeff=0.7, show_results=True)
  image_segmented = mv_1.segment_grabcut(img_str='images\\messi5.jpg', rect_selection_foreground=(50, 50, 450, 290), n_iter=5, show_results=True)


def test_feature_extraction_methods(mv_1):
  '''Test featue extraction methods, e.g. edge detection, corner detection etc.
  '''

  # Edge detection
  # --------------
  edges = mv_1.canny_edge_detection('images\\messi5.jpg', show_results=True)

  # Corner detection
  # ----------------
  images = mv_1.harris_corner_detector('images\\chessboard.png', show_results=True)
  mv_1.harris_corner_detector_subpixel_accuracy('images\\chessboard.png')
  corners_shi_tomasi = mv_1.shi_tomasi_corner_detector(img_str='images\\blox.jpg', max_corners=25, quality_level=0.01, min_distance=10, show_results=True)
  keypoints_sift, descriptors_sift = sift_feature_detector = mv_1.sift_feature_detector(img_str='images\\home.jpg', show_results=True)
  keypoints_surf, descriptors_surf = mv_1.surf_feature_detector(img_str='images\\fly.jpg', hessian_threshold=400, upright=True, show_results=True)
  keypoints_fast = mv_1.fast_feature_detector(img_str='images\\blox.jpg', threshold=10, nonmax_suppresion=True, type='9 16', show_results=True)
  keypoints_censure_brief, descriptors_censure_brief = mv_1.censure_with_brief(img_str='images\\blox.jpg', show_results=True)
  keypoints_orb, descriptors_orb = mv_1.orb_feature_detector(img_str='images\\blox.jpg', show_results=True)
  keypoints_query, keypoints_train, descriptors_query, descriptors_train, matches = mv_1.feature_matching(img_str_1='images\\box.png', img_str_2='images\\box_in_scene.png', type='FLANN WITH SIFT', find_homography_flann=True, min_match_count=10, show_results=True)


def test_enhanced_calibration_methods(mv_1):
  '''Test enhanced calibration methods, e.g. simple camera calibration, computational photography'''

  # Camera calibration
  # ------------------
  mtx, dist, rvecs, tvecs = mv_1.calibrate_camera(folder_img='images\\chess\\', img_str_sample='images\\chess\\left12.jpg', show_results=True)
  rvecs, tvecs = mv_1.calculate_pose(folder_img='images\\chess\\', img_str_sample='images\\chess\\left12.jpg', show_results=True)
  lines_1, lines_2 = mv_1.find_epilines(img_str_1='images\\myleft.jpg', img_str_2='images\\myright.jpg', show_results=True)
  disparity = mv_1.get_depth_map(img_str_1='images\\tsukuba_l.png', img_str_2='images\\tsukuba_r.png', num_disparities=16, block_size=15, show_results=True)

  # Test computational photography methods
  # --------------------------------------
  image_denoised = mv_1.denoise_non_local_means(img_str='images\\die.png', show_results=True)
  image_inpainted = mv_1.inpaint(img_str='images\\yoda_bad_quality.jpg', img_str_mask='images\\yoda_mask.png', method='TELEA', inpaint_radius=3, show_results=True)
  image_hdr = mv_1.high_dynamic_range(folder_img='images\\st_louis_arch\\', method='MERTENS', show_results=True)


def test_ml_methods(mv_1):
  '''Test ML-based methods, e.g. classification, clustering, quantization etc.'''

  class_predicted = mv_1.knn_classificator(train_data=np.random.randint(0, 100,(25,2)).astype(np.float32), 
                                                label_data=np.random.randint(0, 2,(25,1)).astype(np.float32), 
                                                prediction_data=np.random.randint(0, 100,(1,2)).astype(np.float32), 
                                                n_knn=3, show_results=True)
  
  class_predicted = mv_1.ocr_knn_hand_written_digits(img_str='images\\digits.png', n_knn=5, show_results=True)
  class_predicted = mv_1.ocr_svm_hand_written_digits(img_str='images\\digits.png', show_results=True)

  labels, centers = mv_1.kmeans_clustering(train_data=np.hstack(([np.random.randint(25, 100, 25), np.random.randint(175, 255, 25)])), 
                                                             type_criteria='EPS ITER',
                                                             max_iter=10,
                                                             eps=1.0,
                                                             attempts=10,
                                                             n_clusters=2,
                                                             show_results=True)
  labels, centers = mv_1.kmeans_clustering(train_data=np.vstack(([np.random.randint(25, 50, (25, 2)), np.random.randint(60, 85, (25, 2))])), 
                                                                      type_criteria='EPS ITER',
                                                                      max_iter=10,
                                                                      eps=1.0,
                                                                      attempts=10,
                                                                      n_clusters=2,
                                                                      show_results=True)
  
  image_quantized = mv_1.color_quantization(img_str='images\\home.jpg', 
                                                type_criteria='EPS ITER',
                                                max_iter=10,
                                                eps=1.0,
                                                attempts=10,
                                                n_clusters=8,
                                                show_results=True)


def test_dl_methods(mv_1):
  '''Test DL-based methods, e.g. object detection, classification etc.'''
  
  # Object detection using YOLO v8
  # ------------------------------
  mv_1.object_detection_yolo_v8(retrain=False, path_predict_image='images\\cat_dog.jpg')
  mv_1.object_detection_yolo_v8(retrain=True, path_predict_image='images\\red_traffic_light.jpg', path_yaml_file="road_signs\\data.yaml", n_epochs_train=30, optimizer="Adam")
  mv_1.object_detection_yolo_v8(retrain=False, model="yolov8m.pt", path_predict_image="datasets\\coco8\\\images\\test", path_yaml_file="datasets\\coco8.yaml", n_epochs_train=30, optimizer="Adam", batch=4, lr0=1e-3)
  mv_1.object_detection_yolo_v8(retrain=True, model="yolov8m.pt", path_predict_image="datasets\\coco8\\\images\\test\\1 (18).jpg", path_yaml_file="datasets\\african-wildlife.yaml", n_epochs_train=30, optimizer="Adam", batch=4, lr0=1e-3)

  # Classification
  # --------------
  mv_1.classification_dl(data_dir="datasets\\satellite_image_classification", 
                         sz_image=256, color_mode='rgb', label_mode='categorical', 
                         fraction_train=0.7, fraction_validation=0.2, fraction_test=0.1, 
                         dl_method='cnn', 
                         dl_parms_dict={'type_cnn': 1, 'activation_cnn_layer': 'relu', 'activation_cnn_out': 'softmax'}, 
                         optimizer='adam', n_batch=32, n_epoch=300, learning_rate=1e-5, n_patience=5, 
                         pretrain_or_load=False, early_stopping=False, save_model=True, export_training_performance_image=True, export_confusion_matrix_image=True)


def test_hardware_based_methods(mv_1):
  '''Test HW-based methods, e.g. Basler camera handling.'''

  # Hardware-based applications
  #----------------------------
  # Define parameters of Basler camera
  parms = {'mode': 'foreground_loop',
           'n_images': 1, 
           'timeout_ms': 5000, 
           'show_stats': True, 
           'idx_device': 0, 
           'show_one_image': True, 
           'idx_show_one_image': 0}

  # Configure, connect, start and perform image acquisition and dissconnect
  camera, devices = mv_1.basler_camera_configuration(parms)
  camera = mv_1.basler_camera_connect(camera)
  
  while True:
    images = mv_1.grab_software_triggered_camera_basler(parms, camera, 
                                                        devices, show_image=True)
    ans = input("New cycle? y/n")
    if ans=='n':
      break

  camera = mv_1.basler_camera_disconnect(camera)

#=======================================================================================================
if __name__ == '__main__':
   
  # TEMPORARY: Tests
  #================================================================================
  #  Create object
  mv_1 = MV()

  # Test image processing methods
  # test_image_processing_methods(mv_1)

  # Test feature extraction methods
  # test_feature_extraction_methods(mv_1)

  # Test enhanced calibration methods
  # test_enhanced_calibration_methods(mv_1)

  # Test ML-based methods
  # test_ml_methods(mv_1)

  # Test DL-based methods
  # test_dl_methods(mv_1)

  # Test HW-based methods
  test_hardware_based_methods(mv_1)

  pass