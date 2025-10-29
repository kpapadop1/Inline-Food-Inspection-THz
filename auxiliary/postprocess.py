# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script containes the postprocessing functions for the classification 
  task.
  
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script containes the postprocessing functions for the classification 
  task.


SEE ALSO
  -
  
FILE
  .../postprocess.py

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

#==============================================================================
#%% FUNCTION DEFINITIONS
#==============================================================================

#------------------------------------------------------------------------------
# FUNCTION: calculate_multiclass_metrics
def calculate_multiclass_metrics(confusion_matrix):
    pass

# FUNCTION: plot_history
def plot_history(hist, name_computation, export_figure, date_time):
    
    # Show performance: loss function
    fig, axs = plt.subplots(1,2)
    fig_man = plt.get_current_fig_manager()
    fig_man.canvas.manager.set_window_title(name_computation)
    axs[0].plot(hist.history['loss'], color='teal', label='Training', )
    axs[0].plot(hist.history['val_loss'], color='orange', label='Validation')
    axs[0].set_title('Loss', fontsize=20)
    axs[0].legend(loc="upper left")
    axs[0].set(xlabel='Epoch', ylabel='Value')
    axs[0].grid(visible='true')
    
    # Show performance: accuracy
    axs[1].plot(hist.history['accuracy'], color='teal', label='Training', )
    axs[1].plot(hist.history['val_accuracy'], color='orange', label='Validation')
    axs[1].set_ylim(0.17, 1.0)
    axs[1].set_title('Accuracy', fontsize=20)
    axs[1].legend(loc="upper left")
    axs[1].set(xlabel='Epoch', ylabel='Accuracy')
    axs[1].grid(visible='true')
    
    # Export image if desired
    if export_figure:
        name_computation_export = name_computation.replace("/", "_")
        name_computation_export = name_computation_export.replace(".", "")
        plt.savefig("results/loss_accuracy_" + name_computation_export + "_" + 
                    date_time + ".jpg")
    
#------------------------------------------------------------------------------
# FUNCTION: plot_confusion_matrix
def plot_confusion_mat(hist, name_computation, y_test_sparse, 
                       y_test_pred_sparse, export_figure, date_time):
    
    # Get confusion matrix based on test dataset and plot
    print("Set up confusion matrix...")
    confusion = tf.math.confusion_matrix(labels=y_test_sparse, 
                                         predictions=y_test_pred_sparse)
    print(confusion)
    #mat_confusion = confusion.numpy()/np.max(confusion.numpy());
    confusion_mat = confusion_matrix(y_true=y_test_sparse, 
                                     y_pred=y_test_pred_sparse)
    confusion_mat_disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat) 
    confusion_mat_disp.plot(cmap='Blues')
    fig_man = plt.get_current_fig_manager()
    fig_man.canvas.manager.set_window_title(name_computation)

    # Export image if desired
    if export_figure:
        name_computation_export = name_computation.replace("/", "")
        name_computation_export = name_computation_export.replace(".", "")
        plt.savefig("results/confusion_matrix_" + name_computation_export + "_" + 
                    date_time + ".jpg")
        
    return confusion_mat