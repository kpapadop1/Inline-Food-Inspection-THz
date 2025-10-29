# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script defines the global class for FMCW-MIMO super class.
  
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script defines the global class for FMCW-MIMO super class.


SEE ALSO
  -
  
FILE
  .../Fmcw.py

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
Beamforming methods:
- Reconstruction and control of antenna pattern for ULA, URA (or other patterns) based on array factor and element pattern to form antenna pattern
- Conventional and adaptive beamformer (e.g. Minimum Variance Distortionless Response (MVDR; also called Capon Beamformer))
- Beamforming simulator for data generation
- All beamforming methods for phased arrays and digital beamformers (MIMO arrays and hybrid arrays)
- Multifunction Phased Array Radar (MPAR)

-------------------------------------------------------------------------------
"""
#==============================================================================
#%% DEPENDENCIES
#==============================================================================
import os
import numpy as np
import json
import math
import scipy as sp
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point
from shapely.ops import unary_union
from mpl_toolkits.mplot3d import Axes3D
import matlab.engine
from collections import defaultdict
import time
# import cmath
# import copy
# import glob

#==============================================================================
#%% CLASS DEFINITION
#==============================================================================

class Fmcw:
  """General FMCW class containing methods for data import and structuring, processing (e.g. classical FMCW techniques, SAR, Microdoppler etc.), as well as data export and plotting.

  Attributes:
    #-------------
    # Parameters
    #-------------
    parms_board_:
      board_name: 
        Name of the board
      orientation: 
        Orientation ('azimuth', 'elevation')
      power_tx_antenna: 
        Power of transmitting antenna in dBm
      max_gain: 
        Max gain of TX-RX path in dB
      frequency_ref: 
        Reference frequency in Hz
      distance_tx_ref: 
        Distance of TX array to reference corner (upper left of circuit board) in azimuth and eleevation direction (x, y) in m
      distance_rx_ref:
        Distance of TX array to reference corner (upper left of circuit board) in azimuth and eleevation direction (x, y) in m
      pcb_height:
        PCB height in m
      pcb_width:
        PCB width in m
      angle_antenna_3db_oneside:
        One-sided angle in °
      antenna_dimension_azimuth:
        Antenna dimension in azimuth direction in m
      antenna_dimension_elevation: 
        Antenna dimension in elevation direction in m
      number_chips:
        Total number of chips onboard
      tx_id_pos:
        Dict containing sensor-specific (array-specific) coordinates of assigned TX antennas in multiples of wavelength/2 relativ to the upper left element of the rectangular pattern (rows -> elevation; columns -> azimuth)
      rx_id_pos: 
        Dict containing sensor-specific (array-specific) coordinates of     assigned RX antennas in multiples of wavelength/2 relativ to the upper left element of the rectangular pattern (rows -> elevation; columns -> azimuth)
      sweep_bandwidth_max: 
        Max. sweep bandwidth of radar in Hz
      n_adc_bits: 
        Number of ADC bits
      n_lanes:
        Number of lanes of binary data file
      n_frames:
        Number of frames
      if_bandwidth_max: 
        Maximum IF bandwidth in Hz
      sample_rate_max:
        Maximum sample rate of ADC in sps
      dfe_mode:
        DFE mode: 'complex1', 'real/complex2'

    parms_processing:
      n_fft_range: 
        Number of sampling points of range FFT
      n_fft_doppler:
        Number of sampling points of Doppler FFT
      n_fft_angle:
        Number of sampling points of angle FFT
      apply_windowing_range (bool): 
        Apply windowing to range FFT
      apply_windowing_doppler (bool):
        Apply windowing to Doppler FFT
      apply_windowing_angle (bool): 
        Apply windowing to angle FFT
      max_val_angle_norm:
        Maximum value for angle normalization (IWR 1843)
      sar_method: 
        Selected SAR method: Enhanced Backprojection Algorithm: 'EBPA'; Range Migration Algorithm: 'RMA'; Range-Doppler Algorithm: 'RDA'

    parms_chirp_profile:
      t_idle:
        Idle time of chirp in s
      t_idle_min:
        Minimum idle time of chirp in s
      t_tx_start:
        Time of TX starting prior to idle time end in s
      t_adc_start:
        ADC starting time (referenced to end of idle time) in s
      t_ramp:
        Ramp time in s

    parms_target:
      max_target_dimensions:
        Maximum target dimensions (assuming equal dimensions in azimuth and elevation; in m
      distance_pcb_target:
        Distance between PCB (board) and target
      n_cfar_ca_targets:
        Number of returned targets using CFAR-CA 
      targets_sim_init:
        Initial parameters of targets for simulation in order to generate radar data (validation): RCS, position [x,y,z], velocity [x,y,z] / -
      d_max_des:
        Desired maximum range (required for chirp profile determination) in m 
      d_res_des:
        Desired range resolution (required for chirp profile determination) in m 
      v_max_des:
        Desired maximum velocity (required for chirp profile determination) in m/s
      v_res_des:
        Desired velocity resolution (required for chirp profile determination) in m/s

    parms_show:
      show_parameter_file:
        Show parameter file
      show_calibration_parameters:
        Show calibration parameters
      show_virtual_pattern:
        Show virtual pattern
      show_range_doppler_map:
        Show range-Doppler map
      show_cfar_filtered_range_doppler_map:
        Show CFAR-filtered range doppler map
      show_range_angle_map:
        Show range-angle map
      show_sar:
        Show SAR results

    parms_sar:
      platform_speed:
        Longitudinal speed of target
      target_distance:
        Distance to SAR target
      frame_repetition:
        Frame repitition interval in s

    #----------
    # Constants
    #----------
    C:
      Speed of light in m/s

  Methods:
    preprocessing: 
      Preprocess parameters.
    determine_virtual_antennas:
      Set virtual antenna parameter matrix.
    plot_virtual_pattern:
      Plot results of virtual pattern calculation.
    estimate_frame_time:
      Estimate frame time given idle time and total ramp time.
    estimate_number_of_frames_for_stroke:
      Estimate number of frames for one measurement stroke.
    estimate_megabyte_total:
      Estimate number of total bytes for space requirements and feasibility.
    calculate_time_parameters:
      Calculate time parameters.
    calculate_angle_resolution:
      Calculate angle resolution.
    calculate_sar_resolution:
      Calculate SAR-based resolution (angle) in azimuth and elevation.

    get_distances_rx_tx_to_target:
      Get current distances from TX and RX antenna to target respectively.
    simulate_target_position_velocity:
      Simulate position and velocity of targets.
    simulate_sampled_beat_signal:
      Simulate sampled beat signal.
    db_to_lin:
      Convert dB to linear based on scaling type.
    lin_to_dB:
      Convert linear to dB based on power scaling.
    power_receive_radeq:
      Calculate maximum target distance.
    maximum_range_radeq:
      Calculate maximum target distance based on the evaluated radar equation.

    fftshift_freq_grid:
      Apply FFT, shift and get frequency grid.
    hanning_matlab:
      Compute symmetric hanning window like MATLAB's function 'hanning' does.
    fft_range:
      Apply FFT to radar cube in order to acquire range information.
    fft_doppler:
      Apply FFT to radar cube in order to acquire doppler information.
    fft_angle:
      Apply FFT to radar cube in order to acquire angle information.
    plot_map:
      Plot any map as surface.
    doppler_compensation:
      Apply Doppler compensation.
    doa:
      Apply Direction-of-Arrival estimation procedure.
    calculate_target_point:
      Calculate target point in X,Y,Z-space
    cfar_ca:
      Implements the CFAR-CA algorithm.
    range_doppler:
      Calculate range-Doppler profile.
    group_peaks:
      Group peaks in Range-Doppler maps.

    sar_ebpa:
      Apply Enhanced Backprojection (E-BPA) SAR method for imaging.
    sar_rma:
      Apply Range Migration (RMA) SAR method for imaging.
    reshape_data_sar:
      Reshape SAR data
    convert_multistatic_to_monostatic:
      Yanik and Torlak's method for virtual antenna array phase equalization
    reconstruct_sar_image:
      Reconstruct SAR image
    sar_rda:
      Apply Range Doppler Algorithm (RDA) SAR method for imaging.
    plot_sar:
      This function plots the reflectivity function.

    myspecgram:
      Calculate spectrogram.
    stft_basic:
      Compute a basic version of the discrete short-time Fourier transform (STFT).

    read_data_ti_dca_1000:
      Read binary data file captured by TI DCA1000 module.
    print_parameter_file:
      Print important entries of the JSON file object of the measurement.
    extract_parameter_file:
      Extract parameters from parameter file.
    read_bin_file:
      Read binary data file captured by one device in the TI MMWCAS-RF-EVM. 
    read_data_ti_mmwcas_dsp:
      Read binary data file captured by TI MMWCAS-RF-EVM.
    calibrate_data:
      Calibrate data.
    format_output:
      Format output based on desired output format.
    read_dataset_glasgow_2019_radhar:
      Read dataset of University of Glasgow (2019): Radar Signatures of Human Activities. 

    normalize:
      Normalize data structure between 0 and 1.
    oddnumber:
      Calculate the nearest odd number for a given number.
    hanning_window:
      Create Hanning window.

    range_velocity_angle_computation:
      Apply signal processing techniques for the determination of range, velocity and angle of targets.
    microdoppler_computation:
      Apply signal processing techniques for the determination of the microdoppler map.
    simulation_data_to_radar_cube_mimo:
      Simulate MIMO case.
    sar_processing_chain:
      Apply SAR processing chain.
  """

  # region Constants

  C = sp.constants.c

  # endregion

  # region Methods

  def __init__(self, parms_board=None, parms_processing=None, parms_chirp_profile=None, parms_target=None, 
                parms_show=None, parms_sar=None,
                measurement_folder_path=None, parameter_file_path=None, calibration_file_path=None):
      """Default constructor.
      
      Args:
      ------
      parms_board (dict):
        Board parameters
      parms_processing (dict):
        Processing parameters as a dictionary
      parms_chirp_profile (dict): 
        Chirp profile as a dictionary
      parms_target (dict):
        Target parameters as a dictionary
      parms_show (dict): 
        Plot and print parameters as a dictionary
      measurement_folder_path (str):
        Folder path to measurement data
      parameter_file_path (str):
        Path to parameter file (JSON)
      calibration_file_path (str):
        File path to calibration as string
      """

      # Check and assign
      if (parms_board is not None) and (parms_processing is not None) and (parms_chirp_profile is not None) and (parms_target):

        # Assign variables of board parameters
        self.board_name = parms_board['board_name']
        self.orientation = parms_board['orientation']
        self.power_tx_antenna = parms_board['power_tx_antenna']
        self.max_gain = parms_board['max_gain']
        self.frequency_ref = parms_board['frequency_ref']
        self.distance_tx_ref = parms_board['distance_tx_ref']
        self.distance_rx_ref = parms_board['distance_rx_ref']
        self.pcb_height = parms_board['pcb_height']
        self.pcb_width = parms_board['pcb_width']
        self.angle_antenna_3db_oneside = parms_board['angle_antenna_3db_oneside']
        self.n_chips = parms_board['number_chips']
        self.tx_id_pos = parms_board['tx_id_pos']
        self.rx_id_pos = parms_board['rx_id_pos']
        self.sweep_bandwidth_max = parms_board['sweep_bandwidth_max']
        self.n_adc_bits = parms_board['number_adc_bits']
        self.n_lanes = parms_board['number_lanes']
        self.n_frames = parms_board['number_frames']
        self.sample_rate_max = parms_board['rate_sample_max']
        self.if_bandwidth_max = parms_board['if_bandwidth_max']
        self.coeff_margin_sample_rate = parms_board['coeff_margin_sample_rate']
        self.dfe_mode = parms_board['dfe_mode']

        # Assign variables of processing parameters
        self.n_fft_range = parms_processing['n_fft_range']
        self.n_fft_doppler = parms_processing['n_fft_doppler']
        self.n_fft_angle = parms_processing['n_fft_angle']
        self.apply_windowing_range = parms_processing['apply_windowing_range']
        self.apply_windowing_doppler = parms_processing['apply_windowing_doppler']
        self.apply_windowing_angle = parms_processing['apply_windowing_angle']
        self.max_val_angle_norm = parms_processing['max_val_angle_norm']
        self.sar_method = parms_processing['sar_method']

        # Assign variables of chirp profile parameters
        self.t_idle = parms_chirp_profile['t_idle']
        self.t_idle_min = parms_chirp_profile['t_idle_min']
        self.t_tx_start = parms_chirp_profile['t_tx_start']
        self.t_adc_start = parms_chirp_profile['t_adc_start']
        self.t_ramp = parms_chirp_profile['t_ramp']

        # Assign variables of target parameters
        self.max_target_dimensions = parms_target['max_target_dimensions']
        self.distance_pcb_target = parms_target['distance_pcb_target']
        self.n_cfar_ca_targets = parms_target['n_cfar_ca_targets']
        self.targets_sim_init = parms_target['targets_sim_init']
        self.d_max_des = parms_target['d_max_des']
        self.d_res_des = parms_target['d_res_des'] 
        self.v_max_des = parms_target['v_max_des'] 
        self.v_res_des = parms_target['v_res_des']

        # Assign parameters of show parameters
        self.show_parameter_file = parms_show['show_parameter_file']
        self.show_calibration_parameters = parms_show['show_calibration_parameters']
        self.show_virtual_pattern = parms_show['show_virtual_pattern']
        self.show_range_doppler_map = parms_show['show_range_doppler_map']
        self.show_cfar_filtered_range_doppler_map = parms_show['show_cfar_filtered_range_doppler_map']
        self.show_range_angle_map = parms_show['show_range_angle_map']
        self.show_sar = parms_show['show_sar']

        #Assign parameters of SAR parameters
        self.platform_speed = parms_sar['platform_speed']
        self.target_distance = parms_sar['target_distance']
        self.frame_repetition = parms_sar['frame_repetition']

        # Assign optionally file paths
        self.measurement_folder_path = measurement_folder_path
        self.parameter_file_path = parameter_file_path
        self.calibration_file_path = calibration_file_path

        # Apply preprocessing
        self.preprocessing()

      else:
          raise AttributeError('__init__: Imcomplete parameters passed as arguments during initialization. Further operations aborted.')

  def preprocessing(self):
    """Preprocess parameters.

    Construct antenna patterns, define virtual array patterns if desired, design chirps, blocks and frames and calculate grids for FFT computation and maps.
    
    Args:

    Returns:

    Raises:
    """

    # Frequency and wavelength
    self.wavelength = self.C/self.frequency_ref
    self.spacing_rx = self.wavelength/2

    # Determine virtual antennas
    self.determine_virtual_antennas()
   
    # Design or get chirp profile based on targeted properties
    if self.parameter_file_path is None:
      self.determine_time_parameters('design')
    else:
      self.determine_time_parameters('get')

    # Calculate angle-related parameters while taking MIMO into regard
    self.calculate_angle_resolution()

    # Calculate SAR-related parameters
    self.calculate_sar_resolution()

    # Calculate frequency resolution, frequency grid and range grid for range FFT
    self.freq_res = self.sample_rate/self.n_fft_range
    self.freq_grid = np.arange(0,self.n_fft_range)*self.freq_res
    self.range_grid = self.freq_grid*self.C/self.sweep_slope/2

    # Calculate velocity grid. Due to the chirp sequence, it has to be distinguished between azimuth and elevation since every 1/n_TX_direction chirp is used for doppler calculation (i.e. 9 chirps for one block in azimuth and 4 chirps for one block in elevation).
    self.doppler_grid_azimuth = self.fftshift_freq_grid(self.n_fft_doppler, 
                                                        1/self.t_chirp)/self.n_tx_selected_azimuth  # TODO: All chirps instead of selected?
    self.velocity_grid_azimuth = self.doppler_grid_azimuth*self.wavelength/2
    self.doppler_grid_elevation = self.fftshift_freq_grid(self.n_fft_doppler, 
                                                          1/self.t_chirp)/self.n_tx_selected_elevation # TODO: All chirps instead of selected?
    self.velocity_grid_elevation = self.doppler_grid_elevation*self.wavelength/2

    # Calculate angle grid
    self.angle_grid = np.arcsin(np.linspace(-1, 1, self.n_fft_angle))*180/np.pi

  # endregion

  # region Antenna configuration

  def determine_virtual_antennas(self):
    """Determine virtual antennas based on the TX and RX antenna specification.

    The virtual antennas are constructed at the beginning, as the positions of all possible virtual antennas are calculated. Then, all positions that are occupied by multiple virtual antennas are cleared keeping only one virtual antennas. Based on the maximum covering, the virtual antennas in azimuth and elevation are selected and stored automatically. Plotting of the TX, RX and virtual array and preceding steps is performed if desired.
    
    Args:

    Returns:

    Raises:
    """

    # Determine absolute positions of TX antennas
    self.tx_pos_abs = {key: 
                       [self.distance_tx_ref[0]+self.tx_id_pos[key][0]*self.wavelength/2, 
                        -self.distance_tx_ref[1]-self.tx_id_pos[key][1]*self.wavelength/2] 
                       for key in self.tx_id_pos.keys()}

    # Determine absolute positions of RX antennas
    self.rx_pos_abs = {key: 
                       [self.distance_rx_ref[0]+self.rx_id_pos[key][0]*self.wavelength/2, 
                        -self.distance_rx_ref[1]-self.rx_id_pos[key][1]*self.wavelength/2] 
                       for key in self.rx_id_pos.keys()}
    
    # Determine absolute positions of all possible virtual antennas. Assuming that number of receivers is higher than number of senders.
    self.vx_pos_abs = {}
    if len(self.rx_pos_abs.keys()) > len(self.tx_pos_abs.keys()):

      for i_rx, key_rx in enumerate(self.rx_pos_abs.keys()):
        for i_tx, key_tx in enumerate(self.tx_pos_abs.keys()):
          x_v = np.round((self.tx_pos_abs[key_tx][0] + 
                self.rx_pos_abs[key_rx][0])/2, 7)
          y_v = np.round((self.tx_pos_abs[key_tx][1] + 
                self.rx_pos_abs[key_rx][1])/2, 7)
          self.vx_pos_abs[i_rx*len(self.tx_pos_abs.keys()) + i_tx] = (
          [[x_v, y_v], [key_tx, key_rx]])
    else:
      for i_tx, key_tx in enumerate(self.tx_pos_abs.keys()):
        for i_rx, key_rx in enumerate(self.rx_pos_abs.keys()):
          x_v = np.round((self.tx_pos_abs[key_tx][0] + 
                self.rx_pos_abs[key_rx][0])/2, 7)
          y_v = np.round((self.tx_pos_abs[key_tx][1] + 
                self.rx_pos_abs[key_tx][1])/2, 7)
          self.vx_pos_abs[i_tx*len(self.rx_pos_abs.keys()) + i_rx] = (
          [[x_v, y_v], [key_tx, key_rx]])

    # Search for virtul antennas position that are occupied multiple times and remove until only one is left is not required since only the last assigned pair remains.
    unique_first_lists = set()
    vx_unique = {}
    for key, value in self.vx_pos_abs.items():
        first_list_tuple = tuple(value[0])
        if first_list_tuple not in unique_first_lists:
            unique_first_lists.add(first_list_tuple)
            vx_unique[key] = value

    # Select virtual antennas in azimuth and elevation based on the first ones that are found that correspond to the maximum covering. Sort from left to right for azimuth, up to down for elevation.
    azimuth_dict = {}
    for key, value in vx_unique.items():
        y = value[0][1]
        if y not in azimuth_dict:
            azimuth_dict[y] = []
        azimuth_dict[y].append(value)
    elevation_dict = {}
    for key, value in vx_unique.items():
        x = value[0][0]
        if x not in elevation_dict:
            elevation_dict[x] = []
        elevation_dict[x].append(value)

    longest_length = 0
    for index, inner_list in enumerate(azimuth_dict.keys()):
        if len(azimuth_dict[inner_list]) > longest_length:
            longest_length = len(azimuth_dict[inner_list])
            self.vx_azimuth_selected = azimuth_dict[inner_list]
    self.vx_azimuth_selected = sorted(self.vx_azimuth_selected, 
                                      key=lambda x: x[0][0])
    
    longest_length = 0
    for index, inner_list in enumerate(elevation_dict.keys()):
        if len(elevation_dict[inner_list]) > longest_length:
            longest_length = len(elevation_dict[inner_list])
            self.vx_elevation_selected = elevation_dict[inner_list]
    self.vx_elevation_selected = sorted(self.vx_elevation_selected, 
                                        key=lambda x: x[0][1], reverse=True)

    # Plot TX, RX and unique virtual antennas, pcb and origin
    if self.show_virtual_pattern:
      self.plot_virtual_pattern(vx_unique)

    # Determine number of chips, transmitting and receiving antennas for one chip (device) and for all chips. Determine number of all 
    # selected and sorted unique antennas for azimuth and elevation as well.
    self.n_tx_all = len(self.tx_id_pos.keys())
    self.n_rx_all = len(self.rx_id_pos.keys())
    self.n_tx_chip = int(self.n_tx_all/self.n_chips)
    self.n_rx_chip = int(self.n_rx_all/self.n_chips)
    self.n_virtual_antennas_selected_azimuth = len(self.vx_azimuth_selected)
    self.n_virtual_antennas_selected_elevation = len(self.vx_elevation_selected)

    idx_list_tx_azimuth = []
    idx_list_rx_azimuth = []
    for val in self.vx_azimuth_selected:
       idx_list_tx_azimuth.append(val[1][0])
       idx_list_rx_azimuth.append(val[1][1])
    self.n_tx_selected_azimuth = len(np.unique(idx_list_tx_azimuth))
    self.n_rx_selected_azimuth = len(np.unique(idx_list_rx_azimuth))

    idx_list_tx_elevation = []
    idx_list_rx_elevation = []
    for val in self.vx_elevation_selected:
       idx_list_tx_elevation.append(val[1][0])
       idx_list_rx_elevation.append(val[1][1])
    self.n_tx_selected_elevation = len(np.unique(idx_list_tx_elevation))
    self.n_rx_selected_elevation = len(np.unique(idx_list_rx_elevation))

    # Determine chirp groups. Note, that this is not necessarily a schedule oder sorted according to the virtual antenna positions! In this case, the order of the chirp groups for azimuth computation are swapped to match the boards antenna placements.
    grouped = defaultdict(list)
    for item in self.vx_azimuth_selected:
        last_inner_list = item[-1]
        key = last_inner_list[0]
        value = last_inner_list[1]
        grouped[key].append(value)
    chirp_group_txvx_azimuth = dict(grouped)
    self.chirp_group_txrx_azimuth = dict(reversed(list(chirp_group_txvx_azimuth.items())))

    self.chirp_group_txvx_azimuth = {key: [] for key in self.chirp_group_txrx_azimuth}
    for i in self.chirp_group_txrx_azimuth:
       for k in self.chirp_group_txrx_azimuth[i]:
          for key_l, val_l in enumerate(self.vx_azimuth_selected):
            if i == val_l[1][0] and k == val_l[1][1]:
               self.chirp_group_txvx_azimuth[i].append(key_l)

    grouped = defaultdict(list)
    for item in self.vx_elevation_selected:
        last_inner_list = item[-1]
        key = last_inner_list[0]
        value = last_inner_list[1]
        grouped[key].append(value)
    self.chirp_group_txrx_elevation = dict(grouped)

    self.chirp_group_txvx_elevation = {key: [] for key in self.chirp_group_txrx_elevation}
    for i in self.chirp_group_txrx_elevation:
       for k in self.chirp_group_txrx_elevation[i]:
          for key_l, val_l in enumerate(self.vx_elevation_selected):
            if i == val_l[1][0] and k == val_l[1][1]:
               self.chirp_group_txvx_elevation[i].append(key_l)

    pass

  def plot_virtual_pattern(self, vx_unique):
    '''Plot results of virtual pattern calculation.
    
    Args:
      vx_unique (dict):
        All unique virtual antennas

    Returns:

    Raises:
    '''

    # Get values
    x_tx = []
    y_tx = []
    for value in (self.tx_pos_abs.values()):
        x_tx.append(value[0])
        y_tx.append(value[1])
    x_rx = []
    y_rx = []
    for value in (self.rx_pos_abs.values()):
        x_rx.append(value[0])
        y_rx.append(value[1])

    x_vx_multiple = []
    y_vx_multiple = []
    for value in self.vx_pos_abs.values():
        x_vx_multiple.append(value[0][0])
        y_vx_multiple.append(value[0][1])
    x_vx_unique = []
    y_vx_unique = []
    for value in vx_unique.values():
        x_vx_unique.append(value[0][0])
        y_vx_unique.append(value[0][1])
    x_vx_azimuth_selected = []
    y_vx_azimuth_selected = []
    for value in self.vx_azimuth_selected:
        x_vx_azimuth_selected.append(value[0][0])
        y_vx_azimuth_selected.append(value[0][1])
    x_vx_elevation_selected = []
    y_vx_elevation_selected = []
    for value in self.vx_elevation_selected:
        x_vx_elevation_selected.append(value[0][0])
        y_vx_elevation_selected.append(value[0][1])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Antenna configuration')

    # Plot board boundaries
    rectangle = patches.Rectangle((0.0, -self.pcb_height), self.pcb_width, self.pcb_height, edgecolor='black', facecolor='red', alpha=0.1)
    ax[0,0].add_patch(rectangle)
    rectangle = patches.Rectangle((0.0, -self.pcb_height), self.pcb_width, self.pcb_height, edgecolor='black', facecolor='red', alpha=0.1)
    ax[1,0].add_patch(rectangle)
    rectangle = patches.Rectangle((0.0, -self.pcb_height), self.pcb_width, self.pcb_height, edgecolor='black', facecolor='red', alpha=0.1)
    ax[0,1].add_patch(rectangle)
    rectangle = patches.Rectangle((0.0, -self.pcb_height), self.pcb_width, self.pcb_height, edgecolor='black', facecolor='red', alpha=0.1)
    ax[1,1].add_patch(rectangle)

    # Plot TX, RX, multiple VX antennas
    ax[0,0].scatter(x_tx, y_tx, color='red', alpha=0.5, label='TX')
    ax[0,0].scatter(x_rx, y_rx, color='blue', alpha=0.5, label='RX')
    ax[0,0].scatter(x_vx_multiple, y_vx_multiple, color='green', alpha=0.3, label='VX')
    ax[0,0].legend()
    ax[0,0].set_xlabel('X values', fontweight='bold')
    ax[0,0].set_ylabel('Y values', fontweight='bold')
    ax[0,0].set_title('TX, RX, multiple VX', fontweight='bold')
    ax[0,0].grid()

    # Plot TX, RX, unique VX antennas
    ax[1,0].scatter(x_tx, y_tx, color='red', alpha=0.5, label='TX')
    ax[1,0].scatter(x_rx, y_rx, color='blue', alpha=0.5, label='RX')
    ax[1,0].scatter(x_vx_unique, y_vx_unique, color='green', alpha=0.3, label='VX')
    ax[1,0].legend()
    ax[1,0].set_xlabel('X values', fontweight='bold')
    ax[1,0].set_ylabel('Y values', fontweight='bold')
    ax[1,0].set_title('TX, RX, unique VX', fontweight='bold')
    ax[1,0].grid()
    
    # Plot TX, RX, selected unique VX antennas in azimuth
    ax[0,1].scatter(x_tx, y_tx, color='red', alpha=0.5, label='TX')
    ax[0,1].scatter(x_rx, y_rx, color='blue', alpha=0.5, label='RX')
    ax[0,1].scatter(x_vx_azimuth_selected, y_vx_azimuth_selected, color='green', alpha=0.3, label='VX')
    ax[0,1].legend()
    ax[0,1].set_xlabel('X values', fontweight='bold')
    ax[0,1].set_ylabel('Y values', fontweight='bold')
    ax[0,1].set_title('TX, RX, selected unique VX / azimuth', fontweight='bold')
    ax[0,1].grid()

    # Plot TX, RX, selected unique VX antennas in elevation
    ax[1,1].scatter(x_tx, y_tx, color='red', alpha=0.5, label='TX')
    ax[1,1].scatter(x_rx, y_rx, color='blue', alpha=0.5, label='RX')
    ax[1,1].scatter(x_vx_elevation_selected, y_vx_elevation_selected, color='green', alpha=0.3, label='VX')
    ax[1,1].legend()
    ax[1,1].set_xlabel('X values', fontweight='bold')
    ax[1,1].set_ylabel('Y values', fontweight='bold')
    ax[1,1].set_title('TX, RX selected unique VX / elevation', fontweight='bold')
    ax[1,1].grid()

    # Plot origin
    ax[0,0].scatter(0, 0, facecolors='none', edgecolors='black', s=100.0, label='origin')
    ax[1,0].scatter(0, 0, facecolors='none', edgecolors='black', s=100.0, label='origin')
    ax[0,1].scatter(0, 0, facecolors='none', edgecolors='black', s=100.0, label='origin')
    ax[1,1].scatter(0, 0, facecolors='none', edgecolors='black', s=100.0, label='origin')

    # Open another image to plot TX, RX and unique selected VX antennas in azimuth but with more details
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Antenna configuration')
    offset_annotation = [0.000, 0.001]
    offset_rx = [0.0, 0.0015]
    fontsize = 6

    ax.scatter(x_tx, y_tx, color='red', alpha=0.5, label='TX')
    ax.scatter(x_rx, y_rx, color='blue', alpha=0.5, label='RX')
    ax.scatter(x_vx_azimuth_selected, y_vx_azimuth_selected, color='green', alpha=0.3, label='VX')
    for key, value in self.tx_pos_abs.items():
      ax.annotate(key, 
                  value, 
                  np.array(value) + np.array(offset_annotation),
                  fontsize=fontsize,
                  color='red')
    for key, value in self.rx_pos_abs.items():
      ax.annotate(key, 
                  value, 
                  np.array(value) + np.array(offset_annotation),
                  fontsize=fontsize,
                  color='blue')
    for i in self.vx_azimuth_selected:
      ax.annotate(i[1][1], 
                  i[0], 
                  np.array(i[0]) + np.array(offset_annotation) + np.array(offset_rx),
                  fontsize=fontsize,
                  color='blue')
      ax.annotate(i[1][0], 
                  i[0], 
                  np.array(i[0]) + np.array(offset_annotation) ,
                  fontsize=fontsize,
                  color='red')

    ax.legend()
    ax.set_xlabel('X values', fontweight='bold')
    ax.set_ylabel('Y values', fontweight='bold')
    ax.set_title('TX, RX, selected unique VX / azimuth', fontweight='bold')
    ax.grid()

    plt.show(block=False)
      
  # endregion

  # region Chirping

  def estimate_frame_time(self, n_chirps, n_blocks, t_idle, t_ramp, t_interframe):
    """Estimate frame time given idle time and total ramp time.
    
    Estimate frame time given idle time and total ramp time (TI's notation: ramp end time), chirp number and block number (TI's notation: loop number). Use this function for rough calculations, e.g. when using the LUA script of TIs measurement.
    
    Args:
      n_chirps (int): 
        Number of chirps per block
      n_blocks (int):
        Number of blocks per frame
      t_idle (float):
        Idle time in seconds
      t_ramp (float):
        Ramp time in seconds
      t_interframe (float):
        Interframe time in seconds

    Returns:
      t_frame (float):
        Estimated frame time

    Raises:
    """

    t_frame = n_chirps*(t_idle + t_ramp)*n_blocks + t_interframe

    return t_frame

  def estimate_number_of_frames_for_stroke(self, n_chirps, n_blocks, t_idle, t_ramp, t_interframe, t_stroke):
    """Estimate number of frames for one measurement stroke.
    
    Args:
      n_chirps (int): 
        Number of chirps per block
      n_blocks (int): 
        Number of blocks per frame
      t_idle (float): 
        Idle time in seconds
      t_ramp (float): 
        Ramp time in seconds
      t_interframe (float): 
        Interframe time in seconds
      t_stroke (float): 
        Time for one full stroke in seconds

    Returns:
      n_total_frames (int): 
        Number of frames
    
    Raises:
    """

    t_frame = self.estimate_frame_time(n_chirps=n_chirps, 
                                       n_blocks=n_blocks, 
                                       t_idle=t_idle, 
                                       t_ramp=t_ramp, 
                                       t_interframe=t_interframe)
    n_total_frames = np.ceil(t_stroke/t_frame)

    return n_total_frames

  def estimate_megabyte_total(self, n_devices, n_rx_per_device, n_samples, n_chirps, n_blocks, n_frames, n_byte_per_sample):
    """Estimate number of total bytes for space requirements and feasibility.
    
    Args:
      n_devices (int): 
        Number of devices as integer
      n_rx_per_device (int):
        Number of RX antennas as integer
      n_samples (int):
        Number of samples per chirp as integer
      n_chirps (int): 
        Number of chirps per block
      n_blocks (int): 
        Number of blocks per frame
      n_frames (int): 
        Number of frames
      n_byte_per_sample (int): 
        Number of bytes per sample

    Returns:
      n_megabytes_total (int):
        Total megabytes of sampled data
    
    Raises:
    """
      
    n_samples_per_frame = n_devices*n_rx_per_device*n_samples*n_chirps*n_blocks*n_frames*2
    n_megabytes_total = n_byte_per_sample*n_samples_per_frame*1e-6

    return n_megabytes_total

  def determine_time_parameters(self, mode):
      """determine time parameters.
      
      Calculate additional time parameters. See e.g. https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/978334/iwr6843-chip-loops-and-frame-periodicity-in-mmwave-studio-sensor-configuration-tab

      Args:
        mode (str):
          mode of calculation ('design', 'get')

      Returns:

      Raises:
      """

      if mode == 'design':

        # Calculate required parameters according to desired resolutions
        self.t_chirp = self.wavelength/(4*self.v_max_des)
        self.t_frame = self.wavelength/(2*self.v_res_des)
        self.bandwidth = self.C/(2*self.d_res_des)
        if self.bandwidth > self.sweep_bandwidth_max:
          print(f'Sweep bandwidth is limited by hardware to {self.sweep_bandwidth_max*1e-6} GHz. Please check data in order to guarantee consistency.')
        self.sweep_slope = self.bandwidth/self.t_chirp
        self.sample_rate = int(np.round(2*self.sweep_slope*self.d_max_des/self.C))
        # if self.dfe_mode == 'complex1':                                                                       # TODO: Comment this in after debugging simulation function
        #   sample_rate_max = self.sample_rate_max[0]*self.coeff_margin_sample_rate
        # elif self.dfe_mode == 'real/complex2':
        #   sample_rate_max = self.sample_rate_max[1]*self.coeff_margin_sample_rate
        # if self.sample_rate > np.min([sample_rate_max, self.if_bandwidth_max]):
        #   self.sample_rate = np.min([sample_rate_max, self.if_bandwidth_max])
        #   print(f'Sample rate of ADC limited by hardware ({2*self.sweep_slope*self.d_max_des/self.C} -> {np.min([sample_rate_max, self.if_bandwidth_max])} sps).')
        self.n_samples = int(2**np.floor(math.log2(self.t_chirp*self.sample_rate)))
        self.t_sample = 1/self.sample_rate
        self.t_sampling = self.n_samples/self.sample_rate

        # Assume using all chirps (TDM)
        self.n_chirps = len(self.tx_id_pos)

      elif mode == 'get':

        # Extract parameters of measurement
        with open(self.parameter_file_path, 'r') as file:
            parameters_json = json.load(file)
            self.print_parameter_file(parameters_json)
            (n_frames, n_blocks, n_chirps, \
              n_bits, adc_format, \
              freq_start, t_idle, t_tx_start, \
              t_adc_start, time_ramp_end,  n_samples, \
              slope, sample_rate) = self.extract_parameter_file(parameters_json)

        if np.abs(freq_start - self.frequency_ref)/self.frequency_ref > 0.05:
          raise ValueError('get_chirp_frame_time_bandwidth(): Start frequency differs from frequency used in the measurement file. Please ensure consistency.') 

        # Assign variables
        self.n_adc_bits = n_bits
        self.n_samples = n_samples
        self.n_chirps = n_chirps
        self.n_blocks = n_blocks
        self.n_frames = n_frames
        self.sweep_slope = slope
        self.bandwidth = slope*time_ramp_end
        self.sample_rate = sample_rate
        self.t_idle = t_idle
        self.t_tx_start = t_tx_start
        self.t_adc_start = t_adc_start
        self.t_ramp = time_ramp_end
        self.t_sample = 1/sample_rate
        self.t_sampling = self.n_samples/sample_rate
        self.t_chirp = self.t_idle + self.t_ramp
        self.t_block = self.t_chirp*n_chirps
        self.t_frame = self.t_block*n_blocks

        self.t_ramp_excess = self.t_ramp - self.t_adc_start - self.t_sampling
        if self.t_ramp_excess <= 0:
            print(f'Ramp excess time unfeasible for given ADC start time and given total ramp time.')
        self.t_transmitter_on = self.t_ramp + self.t_tx_start
        if self.t_idle < self.t_idle_min:
          print(f'Chirp idle time set to minimum ({self.t_idle_min} s; error: {(self.t_idle_min - self.t_idle)/self.t_idle_min*100} %).')
          self.t_idle = self.t_idle_min
          
        self.t_chirp = self.t_idle + self.t_ramp

      # Determine block time and number of blocks for azimuth and elevation. Keep in mind, that in the most easiest way, all 12 TX antennas (12 chirps) are used to constitute a full block. If azimuth and elevation configuration variies, adapt this part accordingly.
      self.t_block_azimuth = self.n_chirps*self.t_chirp
      self.n_blocks_azimuth = int(np.floor(self.t_frame/
                                           self.t_block_azimuth))
      
      self.t_block_elevation = self.n_chirps*self.t_chirp
      self.n_blocks_elevation = int(np.floor(self.t_frame/
                                             self.t_block_elevation))
      
      # Determine downtime and duty cycle for for azimuth and elevation
      self.t_frame_down_azimuth = self.t_frame - \
        self.t_block_azimuth*self.n_blocks_azimuth
      self.duty_cycle_azimuth = (self.t_block_azimuth*
                                 self.n_blocks_azimuth/self.t_frame)
      self.active_ramp_duty_cycle_azimuth = self.n_chirps*self.n_blocks_azimuth*self.t_ramp/self.t_frame

      self.t_frame_down_elevation = self.t_frame - \
        self.t_block_elevation*self.n_blocks_elevation
      self.duty_cycle_elevation = (self.t_block_elevation*
                                   self.n_blocks_azimuth/self.t_frame)
      self.active_ramp_duty_cycle_elevation = self.n_chirps*self.n_blocks_elevation*self.t_ramp/self.t_frame
      
      self.t_frame_all = self.t_frame*self.n_frames

  def calculate_angle_resolution(self):
    """Calculate angle resolution.
    
    This function calculates the estimated angle resolution at the bore and edges based on a given targeted object distance.
    
    Args:

    Returns:

    Raises:
    """

    # Calculate angle-related parameters while taking MIMO into regard
    self.angle_max = np.arcsin(self.wavelength/(2*self.spacing_rx))
    if self.orientation == 'azimuth':
      n_virtual_antennas = self.n_virtual_antennas_selected_azimuth
      self.angle_res_edge = self.wavelength/(n_virtual_antennas*self.spacing_rx*np.cos(np.arctan(self.max_target_dimensions/self.distance_pcb_target)))
    elif self.orientation == 'elevation':
      n_virtual_antennas =  self.n_virtual_antennas_selected_elevation
      self.angle_res_edge = self.wavelength/(n_virtual_antennas*self.spacing_rx*np.cos(np.arctan(self.max_target_dimensions/self.distance_pcb_target)))
    else:
       raise ValueError("calculate_angle_resolution(): Unknown orientation.")
    
    self.angle_res_bore = self.wavelength/(n_virtual_antennas*self.spacing_rx)

  def calculate_sar_resolution(self):
    """Calculate SAR-based resolution (angle) in azimuth and elevation.
    
    Args:

    Returns:

    Raises:
    """

    # Calculate SAR-related parameters
    self.antenna_dimension_azimuth = self.n_virtual_antennas_selected_azimuth*self.wavelength/2.0
    self.antenna_dimension_elevation = self.n_virtual_antennas_selected_elevation*self.wavelength/2.0
    self.angle_res_sar_azimuth = self.antenna_dimension_azimuth/2.0
    self.angle_res_sar_elevation = self.antenna_dimension_elevation/2.0

  # endregion

  # region Target simulation

  def get_distances_rx_tx_to_target(self, targets_sim_pos, coordinates_tx_i, coordinates_rx_i, index_target):
    """Get current distances from TX and RX antenna to target respectively.
    
    Get current distances from TX and RX antenna to target respectively. For this, the current target positions are used.

    Args:
      targets_sim_pos (numpy array):
        Simulated target positions (default: None)
      coordinates_tx_i (numpy array):
        Coordinates of TX antenna of current virtual channel (default: None)
      coordinates_rx_i (numpy array): 
        Coordinates of RX antenna of current virtual channel (default: None)
      index_target: 
        Index of current target (default: None)
            
    Returns:
    
    Raises:
    """

    # Determine distances to target
    distances_tx_target = np.zeros((self.n_samples))
    distances_rx_target = np.zeros((self.n_samples))
    for k in range(self.n_samples):
      distances_tx_target[k] = np.linalg.norm(targets_sim_pos[index_target, :, k] - coordinates_tx_i)
      distances_rx_target[k] = np.linalg.norm(targets_sim_pos[index_target, :, k] - coordinates_rx_i)

    return distances_tx_target, distances_rx_target

  def simulate_target_position_velocity(self, t_vec_abs):
    """Simulate position and velocity of targets.

    Simulate position and velocity of targets. For this, the initial positions are added along with the time-based integration of velocities in the corresponding direction.

    Args:
      t_vec_abs (numpy array): Absolute time vector in seconds
      p_init_targets (numpy array):
        Initial positions of targets in meters
      v_init_targets (numpy array): 
        Initial velocity vectors of targets in meters/second

    Returns:
      targets_sim_pos (numpy array):
        Simulated target positions

    Raises:
    """

    # Determine target positions (x,y,z) for given absolute time vector
    targets_sim_pos = np.zeros((len(self.targets_sim_init), 3, len(t_vec_abs)))
    for h in range(len(self.targets_sim_init)):

      p_init_target = self.targets_sim_init[h][1]
      v_init_target = self.targets_sim_init[h][2]

      targets_sim_pos[h, 0, :] = p_init_target[0] + v_init_target[0]*t_vec_abs
      targets_sim_pos[h, 1, :] = p_init_target[1] + v_init_target[1]*t_vec_abs
      targets_sim_pos[h, 2, :] = p_init_target[2] + v_init_target[2]*t_vec_abs
    
    return targets_sim_pos

  def simulate_sampled_beat_signal(self, t_vec_rel, distances_tx_target, distances_rx_target, idx_target):
    """Simulate sampled beat signal.
    
    Simulate sampled beat signal based on signal propagation and reflectivity model for a specific path along for multiple targets.
    
    Args:
      t_vec_rel (numpy array): 
        Relative time vector in seconds
      distances_tx_target (numpy array): 
        Distances of TX antenna to targets of current virtual antenna (pair) in meters
      distances_rx_target (numpy array):
        Distances of RX antenna to targets of current virtual antenna (pair) in meters

    Returns:
      x_IF (numpy array):
        Beat signal vector in W

    """
    
    r_roundtrip = distances_tx_target + distances_rx_target
    f_IF = self.sweep_slope*r_roundtrip/self.C
    phase_IF = 2*np.pi*r_roundtrip/self.wavelength

    P_r = self.power_receive_radeq(self.power_tx_antenna, (distances_tx_target + distances_rx_target)/2, self.targets_sim_init[idx_target][0])
    A_target = np.sqrt(self.power_tx_antenna)*np.sqrt(P_r)
    x_IF = A_target * np.exp(1j*(2*np.pi*f_IF*t_vec_rel + phase_IF))

    return x_IF

  def db_to_lin(self, dB, scaling):
    """Convert dB to linear based on scaling type.
    
    Args:
      dB (float):
        dB-value
      scaling (str):
        Type of scaling ('power','voltage')

    Returns:
      lin (float):
        Linear value

    Raises:
    """

    if scaling == 'power':
      lin = 10**(dB/10)
    elif scaling == 'voltage':
      lin = 10**(dB/20)

    return lin
  
  def lin_to_dB(self, lin, scaling):
    """Convert linear to dB based on power scaling.
    
    Args:
      lin (float):
        Linear valze
      scaling (float): 
        Type of scaling ('power','voltage')

    Returns:
      lin (float):
        dB-value

    Raises:
    """
    
    if scaling == 'power':
      lin = 10*np.log10(lin)
    elif scaling == 'voltage':
      lin = 20*np.log10(lin)
      
    return lin

  def power_receive_radeq(self, power_tx, target_r, target_rcs):
    """Calculate maximum target distance.
    
    Calculate maximum target distance based on the evaluated radar equation.
    
    Args:
      power_tx (float): 
        Power of transmitting antenna in W
      target_r (float):
        Distance to target in m
      target_rcs (float):
        Radar cross-section of target in m²

    Returns:
      power_rx (float):
        Power of receiveing antenna / W

    Raises:
    """

    # Get power in W and calculate receive power / W
    power_tx_lin = self.db_to_lin(power_tx, 'power')*1e-3
    power_rx = power_tx_lin*self.max_gain**2*self.wavelength**2*target_rcs/(np.mean(target_r)**4*(4*np.pi)**3)

    return power_rx

  def maximum_range_radeq(self, power_tx, power_rx, target_rcs):
    """Calculate maximum target distance based on the evaluated radar equation.
    
    Args:
      power_tx (float):
        Power of transmitting antenna in W
      power_rx (float):
        Power of receiveing antenna in W
      target_rcs (float):
        Radar cross-section of target in m²

    Returns:
      r_max_signal (float):
        Maximum distance to target in m

    Raises:
    """
      
    r_max_signal = np.power(power_tx*self.antenna_gain_tx**2*self.wavelength**2*target_rcs/(power_rx*(4*np.pi)**3), 1/4)

    return r_max_signal

  # endregion

  # region Standard MIMO processing

  def fftshift_freq_grid(self, N, Fs):
    """Apply FFT, shift and get frequency grid.

    Args:
      N (int): 
        Number of sampling points
      Fs (float):
        Sampling frequency in Hz

    Returns:
      freq_grid (numpy array):
        Frequency grid in Hz

    Raises:
    """

    freq_res = Fs/N
    freq_grid = np.arange(0, N)*freq_res
    Nyq = Fs/2
    half_res = freq_res/2
    if np.remainder(N, 2): # Odd
        idx = np.arange(1, (N - 1)/2 + 1).astype(int)
        halfpts = int((N + 1)/2)
        freq_grid[halfpts - 1] = Nyq - half_res
        freq_grid[halfpts] = Nyq + half_res
    else:
        idx = np.arange(1, N/2 + 1).astype(int)
        halfpts = int(N/2 + 1)
        freq_grid[halfpts - 1] = Nyq

    freq_grid = sp.fft.fftshift(freq_grid)
    freq_grid[idx - 1] = freq_grid[idx - 1] - Fs

    return freq_grid

  def hanning_matlab(self, n):
      """Compute symmetric hanning window like MATLAB's function 'hanning' does.
      
      Args:
        n (int): Window length

      Returns:

      Raises:
      """

      m = n/2
      w = 0.5*(1.0 - np.cos(2*np.pi*np.arange(1,m + 1)/(n + 1)))
      w = np.hstack((w, w[::-1]))
      return w
  
  def fft_range(self, radar_data_cube_tx):
      """Apply FFT to radar cube in order to acquire range information.
      
      Args:
        radar_data_cube_tx (3D numpy array): 
          Unprocessed transmitter-specific radar data cube of size [n_samples (rows) x n_doppler (columns; n_blocks) x n_channel]

      Returns:
        range_data_cube_tx (numpy array): 
          Transmitter-specific range data cube [n_fft_range (rows) x n_doppler (columns; n_blocks) x n_channel]

      Raises:
      """

      # Get dimensions
      n_range = np.shape(radar_data_cube_tx)[0]
      n_doppler = np.shape(radar_data_cube_tx)[1]
      n_angle = np.shape(radar_data_cube_tx)[2]

      # Apply Range FFT
      range_data_cube_tx = np.zeros((self.n_fft_range, n_doppler, n_angle), dtype=np.complex128)
      for i in range(n_doppler):
        for j in range(n_angle):
          if self.apply_windowing_range:
              range_fft = radar_data_cube_tx[:,i,j]*self.hanning_matlab(n_range)
          else:
              range_fft = radar_data_cube_tx[:,i,j]
          range_data_cube_tx[:,i,j] = np.fft.fft(range_fft, n=self.n_fft_range)

      return range_data_cube_tx
  
  def fft_doppler(self, range_data_cube_tx):
      """Apply FFT to radar cube in order to acquire doppler information.
      
      Based on the MIMO signal processing chain, this is applied to the range-processed radar cube.
      
      Args:
        range_data_cube_tx (3D numpy array): 
          Transmitter-specific range data cube [n_fft_range (rows) x n_doppler (columns; n_blocks) x n_channel]

      Returns:
        doppler_data_cube_tx (3D numpy array): 
          Transmitter-specific range-Doppler data cube [n_fft_range (rows) x n_fft_doppler (columns) x n_channel]

      Raises:
      """

      # Get dimensions
      n_range = np.shape(range_data_cube_tx)[0]
      n_doppler = np.shape(range_data_cube_tx)[1]
      n_angle = np.shape(range_data_cube_tx)[2]

      # Apply Doppler FFT
      doppler_data_cube_tx = np.zeros((n_range, self.n_fft_doppler, n_angle), dtype=np.complex128)
      for i in range(n_range):
          for j in range(n_angle):
              if self.apply_windowing_doppler:
                  doppler_fft = range_data_cube_tx[i,:,j].flatten()*self.hanning_matlab(n_doppler)[0:n_doppler]                   # TODO: Debug, temporary cutoff
              else:
                  doppler_fft = range_data_cube_tx[i,:,j].flatten()
              doppler_data_cube_tx[i,:,j] = np.fft.fftshift(np.fft.fft(doppler_fft, n=self.n_fft_doppler), axes=0)

      return doppler_data_cube_tx

  def fft_angle(self, radar_cube, detected=None):
      """Apply FFT to radar cube in order to acquire angle information. 
      
      Based on the MIMO signal processing chain, 
      this is applied to the range-Doppler-processed radar cube.

      Args:
        doppler_data_cube_tx (3D numpy array):
          Transmitter-specific range-Doppler data cube [n_fft_range x n_receiver x n_fft_doppler]
        detected (2D list):
          List of detected indices. If empty that angle FFT will be applied over the whole RD-processed radar cube (default: None).

      Returns:
        radar_cube_angle (3D numpy array):
          Radar cube after angle processing
      
      Raises:
      """

      # Get dimensions
      n_range = np.shape(radar_cube)[0]
      n_doppler = np.shape(radar_cube)[1]
      n_angle = np.shape(radar_cube)[2]

      angle =  np.zeros((n_range, n_doppler, self.n_fft_angle), dtype=np.complex128)
      for i in range(n_range):
        for j in range(n_doppler):
          if self.apply_windowing_angle:
              angle_fft = radar_cube[i,j,:].flatten()*sp.signal.windows.taylor(n_angle, norm=False)
          else:
              angle_fft = radar_cube[i,j,:].flatten()

          # # Apply phase compensation on the range-velocity bin for virtual  if target was detected
          # matching_element = next((d for d in detected if d['indices'] == (i,j)), None)
          # if matching_element:
          #   pha_comp = np.exp(-1j * np.pi * 
          #                     (int(matching_element['indices'][0]) - self.n_fft_doppler/2) / self.n_fft_doppler)
          #   angle_fft = angle_fft*pha_comp

          angle[i,j,:] = np.fft.fftshift(np.fft.fft(angle_fft, n=self.n_fft_angle))

      return angle

  def plot_map(self, y_grid, x_grid, data=None, x_label=None, y_label=None, z_label=None, title=None, mode=None, extent=None, add_points=None, limits=None):
      """Plot map as surface.
      
      Args:
        y_grid (numpy array): 
          y-grid
        x_grid (numpy array):
          x-grid
        data (2D numpy array)
          Data (default: None)
        x_label (str):
          X-axis label
        y_label (str):
          Y-axis label
        z_label (str):
          Z-axis label
        title (str):
          Title
        mode (str):
          Mode of plotting ('2D'; '3D')
        extent (list):
          Extent of 2D maps
        add_points (list):
          Additional points to plot
        limits (list):
          Limits ([[x_min, x_max], y_min, y_max], z_min, z_max])

      Returns:

      Raises:
      """

      if data is not None:
        data = np.abs(data)
      
      if mode == '2D':

        # Create a 2D plot and plot range map and range-Doppler map
        fig, ax = plt.subplots(nrows=1 , ncols=1, sharex=False, sharey='all')

        if data is not None:
          ax0 = ax.imshow(data, 
                          extent=[x_grid[0], x_grid[-1], 
                                  y_grid[-1], y_grid[0]], cmap='viridis', interpolation='nearest', aspect='auto')
        plt.colorbar(ax0, ax=ax, label='Color scale')
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        ax.set_title(title, fontweight='bold')

        plt.show(block=False)

      elif mode == '3D':

        # Create a 3D plot and plot surface
        fig, ax = plt.subplots(nrows=1 , ncols=1, sharex=False, sharey='all')
        ax.axis('off') 
        ax0 = fig.add_subplot(111, projection='3d')
        fig.canvas.manager.set_window_title(title)

        # Plot the surfaces of range map and range-Doppler map
        x_mesh, y_mesh  = np.meshgrid(x_grid, y_grid)
        if data is not None:
          surf = ax0.plot_surface(x_mesh, 
                                  y_mesh, 
                                  data, 
                                  cmap='viridis')
      
        if add_points is not None:
          if isinstance(add_points, list):
            for i in range(len(add_points)):
              indices = add_points[i]['indices']
              value = add_points[i]['value']
              ax0.scatter(x_grid[indices[0]], 
                          y_grid[indices[1]], 
                          value, 
                          color='red', s=10, label=str(i))
          elif isinstance(add_points, np.ndarray):
            if add_points.ndim == 1:
              for i in range(len(add_points)):
                ax0.scatter(add_points[0], 
                            add_points[1], 
                            add_points[2], 
                            color='red', s=10, label=str(i))
            elif add_points.ndim == 2:
              for i in range(len(add_points)):
                ax0.scatter(add_points[i, 0], 
                            add_points[i, 1], 
                            add_points[i,2], 
                            color='red', s=10, label=str(i))
        ax0.scatter(0.0, 0.0, 0.0, color='black', s=10)
        ax0.view_init(elev=30, azim=-120)
        ax0.set_xlabel(x_label, fontweight='bold')
        ax0.set_ylabel(y_label, fontweight='bold')
        ax0.set_zlabel(z_label, fontweight='bold')
        ax0.set_title(title, fontweight='bold')

        if limits is not None:
          ax0.set_xlim(limits[0])
          ax0.set_ylim(limits[1])
          ax0.set_zlim(limits[2])

        if data is not None:
          fig.colorbar(surf)

        # Show plot
        plt.show(block=False)

  def doppler_compensation(self, range_all, detected):
    """Apply Doppler compensation.
    
    Apply Doppler compensation using the maximum intensity peak on each range bin based on the detected range bins. Do this for all virtual antennas except the first one.
    
    Args:
      range_all (numpy array):
        Range data
      detected (list):
        Detected targets as indices in range and doppler axis

    Returns:
      range_all (numpy array):
        Doppler-compensated range data

    Raises:
    """

    # Get doppler and range indices of detected targets
    detected_idx_doppler = np.array([detected[i]['indices'][0] for i in range(len(detected))])
    detected_idx_range = np.array([detected[i]['indices'][1] for i in range(len(detected))])

    # Apply for every range bin
    for i in range(0, self.n_fft_range):

      # Check if current range bin was marked by the detection algorithm.
      find_idx = np.where(detected_idx_range == i)[0]

      if len(find_idx) == 0:
        continue
      else:
        # Get doppler index based on range bin, where detection ocurred, and apply phase compensation. For this, pick the first
        # larger velocity. Concatenate the transmitter-specific data matrices after that.
        pick_idx = find_idx[0]
        #phase_compensation = np.exp(-1j * np.pi * (detected_idx_doppler[pick_idx] - self.n_fft_doppler/2) / (self.n_fft_doppler*self.n_tx_all))
        phase_compensation = np.exp(-1j * np.pi * (detected_idx_doppler[pick_idx] - self.n_fft_doppler/2) / self.n_fft_doppler)
        range_all[i, 1:, :] = range_all[i, 1:, :]*phase_compensation

    return range_all

  def doa(self, detected, range_all):
      """Apply Direction-of-Arrival estimation procedure.
      
      Args:
        detected (list):
          Detected targets as indices in range and doppler axis
        range_all (3D numpy array):
          Doppler-compensated range data

      Returns:
        angle_crop (3D numpy array):
         Angle matrix or angle ffts on detected targets
      
      Raises:
      """

      # Apply angle FFT on all positions and plot if desired
      doppler_all = self.fft_doppler(range_all)
      angle_fft = self.fft_angle(doppler_all, detected)

      return angle_fft

  def calculate_target_point(self, angle_fft, detected):
      """Plot range angle map.

      Args:
        angle matrix (3D numpy array):
          Processed radar cube after angle FFT
        detected (list):
          Detected targets as indices in range and doppler axis

      Returns:
        target_xyz (list):
          Target coordinates (x,y,z)

      Raises:
      """

      target_angle_index = np.unravel_index(np.argmax(angle_fft[:, detected[0]['indices'][0]]), angle_fft[:, detected[0]['indices'][0]].shape)
      
      target_angle = self.angle_grid[int(target_angle_index[1])]
      x = (self.range_grid[detected[0]['indices'][0]]*
          np.sin(target_angle*np.pi/180))
      z = 0.0
      y = (self.range_grid[detected[0]['indices'][0]]*
          np.cos(target_angle*np.pi/180))
      target_xyz = np.array([x, y, z])
      
      return target_xyz

  def cfar_ca(self, rdm, orientation):
      """Implements the CFAR-CA algorithm.
      
      Args:
        rdm (numpy array): 
          Range-Doppler map
        orientation (str):
          Orientation ('azimuth': Azimuth; 'elevation': Elevation)

      Returns:
        idx_rdm_target_detected (numpy array):
          Indices of target detection
        rdm_cfar_ca (numpy array):
          Range-Doppler map after filtering

      Raises:
      """
      
      self.cfar_t_r = 2
      self.cfar_g_r = 1
      self.cfar_t_d = 2
      self.cfar_g_d = 1
      self.cfar_offset = 0.1

      # CFAR-based target detection
      #----------------------------
      # Design a loop such that it slides the CUT across range-Doppler map by giving margins at the edges for training and guard Cells. For every iteration sum the signal level within all the training cells. To sum convert the value from logarithmic to linear using db2pow function. Average the summed values for all of the training cells used. After averaging convert it back to logarithmic using pow2db. Further add the offset to it to determine the threshold. Next, compare the signal under CUT with this threshold. If the CUT level > threshold assign it a value of 1, else equate it to 0.

      # Determine number of doppler cells on either side of CUT, number of range cells on either side of CUT, number of range dimension cells and number of doppler dim. cells
      radius_doppler = self.cfar_t_d + self.cfar_g_d
      radius_range = self.cfar_t_r + self.cfar_g_r
      n_range_cells = self.n_fft_range - 2*radius_range
      n_doppler_cells = self.n_fft_doppler - 2*radius_doppler

      rdm_cfar_ca = np.zeros_like(rdm)
      threshold = np.zeros_like(rdm)
      r_min = radius_range
      r_max = n_range_cells - radius_range - 1
      d_min = radius_doppler
      d_max = n_doppler_cells - radius_doppler - 1
      noise_level = np.zeros_like(rdm)
      max_val = np.max(rdm)
      
      # Pre-compute constants outside the loop
      max_val_inv = 1 / max_val
      guard_mask_shape = (2 * radius_range + 1, 2 * radius_doppler + 1)
      guard_mask = np.ones(guard_mask_shape, dtype=bool)
      guard_mask[radius_range-self.cfar_g_r:radius_range+self.cfar_g_r + 1, 
                radius_doppler-self.cfar_g_d:radius_doppler+self.cfar_g_d + 1] = False

      for r in range(r_min, r_max + 1):
          for d in range(d_min, d_max + 1):
              # Define the training cells window
              training_cells = rdm[r-radius_range:r+radius_range+1, 
                                  d-radius_doppler:d+radius_doppler+1]

              # Calculate the average noise level using the training mask
              training_scaled = training_cells[guard_mask] * max_val_inv
              noise_level[r, d] = 10 * np.log10(
                 np.mean(10**(training_scaled / 10)))
              threshold[r, d] = noise_level[r, d] + self.cfar_offset
              
              # Determine if the cell under test exceeds the threshold
              if rdm[r, d] * max_val_inv >= threshold[r, d]:
                  rdm_cfar_ca[r, d] = rdm[r, d]

      # Return the indices of the first n targets after the loops
      indices = np.argpartition(rdm_cfar_ca.ravel(), -self.n_cfar_ca_targets)[-self.n_cfar_ca_targets:]
      idx_rdm_target_detected = np.unravel_index(indices, rdm_cfar_ca.shape)

      return idx_rdm_target_detected, rdm_cfar_ca

  def range_doppler(self, radar_data_cube, mode, orientation, idx_frame=0, idx_channel_rd_plot=0):
      """Calculate range-Doppler profile.
      
      Args:
        radar_data_cube (3D numpy array):
          Data of one frame of following dimensions: [n_samples, n_receiver, n_virtual_antennas_unique]
        mode (str):
          Either 'range-velocity-angle' (plots afterwards the range-Doppler map) or 'microdoppler' (RD map plotting is surpressed)
        orientation (str):
          Orientation ('azimuth': Azimuth; 'elevation': Elevation)
        idx_frame (int):
          Index of frame (default: 0)
        idx_channel_plot (int):
          Index of channel for RD plot (default: 0)

      Returns:
        range_data (3D numpy array):
          Range data processed out of raw radar cube
        doppler_data (3D numpy array): 
          Range-Doppler data processed out of range data
        doppler_data_sum (3D numpy array):
          Summed up Range-Doppler data

      Raises:
      """
    
      # Apply FFTs to get range and velocity distribution
      range_data = self.fft_range(radar_data_cube)
      doppler_data = self.fft_doppler(range_data)
      doppler_data_sum = np.mean(np.abs(doppler_data), 2)

      return range_data, doppler_data, doppler_data_sum

  def group_peaks(self, rdm, size_percent_max=5, orientation='azimuth'):
    """Group peaks in Range-Doppler maps.
    
    Args:
      rdm (2D numpy array):
        Range-Doppler map
      size_percent_max (int):
        Filter size (default: 5)
      orientation (str): 
        Orientation ('azimuth': Azimuth; 'elevation': Elevation)

    Returns:
      detected (List):
        List of detected targets (elements: indices, value)
      rdm (numpy array):
        Peak-processed Range-Doppler map

    Raises:
    """

    # Determine size for filter using the maximum size percentage (of range and doppler direction)
    size = np.ceil((np.max([size_percent_max/100*self.n_fft_range, size_percent_max/100*self.n_fft_doppler])))
  
    # Apply maximum filter to find local maxima
    filtered = sp.ndimage.maximum_filter(rdm, size=size)

    # Create a binary mask of the peaks and apply to get a subset, which is used to label
    peaks = (rdm == filtered)
    labeled_peaks, num_features = sp.ndimage.label(peaks)
    
    detected = []
    for region in range(1, num_features + 1):
      region_mask = (labeled_peaks == region)
      max_value = rdm[region_mask].max()
      if max_value > 0.0:
        y_indices, x_indices = np.where(region_mask)
        centroid = (int(np.mean(x_indices)), int(np.mean(y_indices)))
        detected.append({'indices': centroid, 'value': max_value})

    return detected, rdm

  # endregion 

  # region SAR processing

  # TODO: Validation
  def sar_ebpa(self, s):
    """Apply Enhanced Backprojection (E-BPA) SAR method for imaging.
    
    Args:
      s (numpy array): 
        Signal data matrix s(x', y_R, y_T, k)
    
    Returns:
      p (numpy array):
        Reflectivity matrix

    Raises:
    """

    # Get data s(x', y_R, y_T, k) and apply Fourier transform (convolutional theorem 
    # applied in order to prevent convolution and replacing it with the product S*H)
    S = np.fft.fft(s, axis=0)

    # TODO: Temporary!
    x = np.arange(0,9)
    y = np.arange(10,17)
    z = np.arange(20,40)
    Delta_x = 0.001
    y_T = 0.04*np.arange(0,11)
    y_R = 0.08*np.arange(0,12)
    z_0 = 0.3
    k = 1000*np.arange(0,13)

    # Define matched filter h(x, y, z, y_T, y_R, k) and compute its Fourier transform
    x_reshaped = x[:, np.newaxis,  
                    np.newaxis, np.newaxis,  
                    np.newaxis, np.newaxis]
    y_reshaped = y[np.newaxis, :,  
                    np.newaxis, np.newaxis,  
                    np.newaxis, np.newaxis]
    z_reshaped = z[np.newaxis, np.newaxis, 
                    :, np.newaxis,  
                    np.newaxis, np.newaxis]
    y_T_reshaped = y_T[np.newaxis, np.newaxis, 
                        np.newaxis, :,  
                        np.newaxis, np.newaxis]
    y_R_reshaped = y_R[np.newaxis, np.newaxis, 
                        np.newaxis, np.newaxis,  
                        :, np.newaxis]
    k_reshaped = k[np.newaxis, np.newaxis, 
                    np.newaxis, np.newaxis,  
                    np.newaxis, :]
    h = np.exp(1j*k_reshaped*np.sqrt((x_reshaped - Delta_x/2)**2 + (y_reshaped - y_T_reshaped)**2 + (z_reshaped - z_0)**2)) * \
        np.exp(1j*k_reshaped*np.sqrt((x_reshaped + Delta_x/2)**2 + (y_reshaped - y_R_reshaped)**2 + (z_reshaped - z_0)**2))
    H = np.fft.fft(s, axis=0)

    # Solve triple integral
    S_reshaped = S[:, np.newaxis, np.newaxis, :, :, :]
    P = np.sum(S_reshaped*H, axis=(-3, -2, -1))

    # Apply inverse Fourier transformation along k_x wavenumber domain
    p = np.fft.ifft(P, axis=0)

    return p

  # TODO: Validation
  def sar_rma_temp(self, s):
    """Apply Range Migration (RMA) SAR method for imaging.
    
    Args:
      s (numpy array): 
        Signal data matrix s(x', y_R, y_T, k)

    Returns:
      p (numpy array): 
        Reflecitivity function p(x, y, z)

    Raises:
    """

    # TODO: Temp.
    x = np.arange(0,9)
    y = np.arange(10,17)
    k = 1000*np.arange(0,13)
    #z = np.arange(20,40)
    Delta_x = 0
    Delta_y = 0.001
    z_0 = 0.3

    # Get data s(x', y_R, y_T, k) and convert from multistatic to monostatic aperture: s(x', y', k)
    e_term_1 = np.exp(-1j*k*(Delta_x**2 + Delta_y**2)/(4*z_0))
    e_term_1 = e_term_1[np.newaxis, np.newaxis, np.newaxis, :]
    s_virtual = s*e_term_1

    # Determine spatial frequencies required for equidistant k_z determination
    k_x =  2*np.pi*np.fft.fftfreq(n=s_virtual.shape[0], d=Delta_x)
    k_y =  2*np.pi*np.fft.fftfreq(n=s_virtual.shape[1], d=Delta_y)
    k_z = np.sqrt(4*k**2 - k_x**2 - k_y**2)

    # Apply 2D Fourier-Transform
    S_virtual_k_x = np.fft.fft(s_virtual, axis=0)
    S_virtual_k_x_k_y = np.fft.fft(S_virtual_k_x, axis=1)
    
    # Resample S(k_x,k_y,k) to get S(k_x,k_y,k_z) 
    S_virtual_resampled = np.zeros((s_virtual.shape[0], s_virtual.shape[1], len(k_z)))
    for i in range(s_virtual.shape[0]):
      for j in range(s_virtual.shape[1]):
        S_virtual_resampled[i, j, :] = np.interp(S_virtual_k_x_k_y[i, j, :], k, k_z)

    # Determine reflectivity function
    e_term_2 = np.exp(-1j*k_z*z_0)
    e_term_2 = e_term_2[np.newaxis, np.newaxis, :]
    P = e_term_2*S_virtual_resampled

    # Calculate reflectivity function
    p = np.fft.ifft(np.fft.ifft(np.fft.ifft(P, axis=0), axis=1), axis=2)
    
    return p

  def sar_rma(self, s):
    """Apply Range Migration (RMA) SAR method for imaging.
    
    Args:
      s (numpy array): 
        Signal data matrix s(x', y_R, y_T, k)

    Returns:
      p (numpy array): 
        Reflecitivity function p(x, y, z)

    Raises:
    """
    # Define paramters

    self.frequency = [self.frequency_ref, 
                      self.sweep_slope,
                      self.sample_rate,
                      self.t_adc_start]
    self.n_horizontal_scan = len(s)
    self.x_step_mm = self.platform_speed*self.frame_repetition*1000
    self.y_step_mm = (self.wavelength/4)*1e3
    
    #Temp.: Test sorted Data
    # Sort adc_out for MIMO RMA
    raw_data_sorted = self.reshape_data_sar(s)
    
    # Convert multistatic data to monostatic data
    sar_data_monostatic = self.convert_multistatic_to_monostatic(raw_data_sorted,self.frequency)

    # Make uniform virtual array
    raw_data_uniform = np.reshape(sar_data_monostatic,(-1, self.n_horizontal_scan, self.n_samples), order='F')


    # HACK
    # Plot the SAR image as a 2D mesh
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(np.abs(raw_data_uniform[:,0,:]))
    plt.show(block=False)
    
    sar_image = self.reconstruct_sar_image(raw_data_uniform, 
                                           self.frequency, 
                                           self.x_step_mm, 
                                           self.y_step_mm,
                                           self.target_distance*1e3, 
                                           [512, 256]) # [256, 128]
    
    return sar_image

  def reshape_data_sar(self, s):
    """Sorts the read-in data into the required format 
    
    The read-in data is converted into the format (n_vitual_channel, y_point_m, x_point_m, n_samples)
    This enables the SAR image to be analysed directly for all virtual channels across all frames.

    This method should only be used if the number of chirps per frame is equal to 1 so that y_point_m is set to one in case of one dimensional scanning.     
    
    Args:
      raw_data_calibrated (list):
        list containing calibrated raw data
    
    Returns:
      reshaped_out_data (numpy array):
        array containing calibrated data
        
    Raises:
    """
    # stack arrays vertical
    out_data = np.stack(s, axis=0)

    # transpose axes to get format (n_virtual_channel, n_vertical_scan, n_horizontal_scan, n_samples) only valid if n_blocks is 1, otherwise the data has to be sorted in another way
    reshaped_out_data = np.transpose(out_data,(3, 2, 0, 1))


    # # HACK-----------------
    # i_frame = 0
    # i_block = 1
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    # fig.canvas.manager.set_window_title(f"Frame: {i_frame} / Block: {i_block}")

    # ax[0,0].plot(reshaped_out_data[0:26, i_frame, i_block, :].T)
    # ax[0,0].set(xlabel="sample", ylabel="value", title="VX 0-25")
    # ax[0,0].legend()

    # ax[0,1].plot(reshaped_out_data[26:51, i_frame, i_block, :].T)
    # ax[0,1].set(xlabel="sample", ylabel="value", title="VX 26-50")
    # ax[0,1].legend()

    # ax[1,0].plot(reshaped_out_data[51:76, i_frame, i_block, :].T)
    # ax[1,0].set(xlabel="sample", ylabel="value", title="VX 51-75")
    # ax[1,0].legend()

    # ax[1,1].plot(reshaped_out_data[76:85, i_frame, i_block, :].T)
    # ax[1,1].set(xlabel="sample", ylabel="value",title="VX 76-85")
    # ax[1,1].legend()

    # plt.show(block=False)
    # pass
    # # HACK-----------------


    # Temp for one block
    reshaped_out_data = reshaped_out_data[:, 0:1, :, :]

    return reshaped_out_data
     
  def convert_multistatic_to_monostatic(self, sar_data_multistatic, frequency):
    """Convert multistatic to monostatic. 
    This Method convertes multistatic data to monostatic data.
    Format of multistatic data has to be n_channel x y_point_m x x_point_m x n_sample

    Args:
      sar_data_multistatic(arr):
        array containing multistatic data
      frequency(numpy array):
        array containing frequency_ref (Hz),sweep_slope (Hz/s),sample_rate(sps),t_adc_start(s):
      x_step_mm (int):
        step size in x axis (horizontal) in mm
      y_step_mm (int):
        step size in y axis (vertical) in mm
      z_target_mm (int):
        target distance in mm
      n_tx_active (int):
        number of active tx antennas
      n_rx_antenns (int):
        number of active rx antennas
    
    Returns:
      sar_data_monostatic (numpy array):
        array containing converted sar_data
    
    Raises:
    """
    # Define frequency spectrum
    n_channel, y_point_m, x_point_m, n_sample = sar_data_multistatic.shape
    if (len(self.frequency) > 1) and (len(self.frequency) <= 4) and (n_sample > 1):
        f0, K, fS, adcStart = frequency
        f0 = f0 + adcStart*K; # This is for ADC sampling offset
        f = f0 + np.arange(0, n_sample)*K/fS; # wideband frequency
    elif (len(frequency)==1) and (n_sample == 1):
        f = frequency
    else:
        print('Please correct the frequency configuration and data.')

    # Define fixed parameters
    k = 2*np.pi*f/self.C
    k = np.reshape(k,(1,1,1,-1), order='F')
    
    # Get virtual antenna positions
    n_rx = self.n_rx_selected_azimuth
    n_tx = self.n_tx_selected_azimuth
  
    # Define measurement locations at linear rail (coordinates: [x y z], x-horizontal, y-vertical, z-depth)
    # only valid if n_blocks is 1, otherwise the data has to be sorted in another way
    x_axis_m = self.x_step_mm * np.arange(-(x_point_m - 1)/2, (x_point_m - 1)/2 + 1) * 1e-3
    y_axis_m = self.y_step_mm * np.arange(-(y_point_m - 1)/2, (y_point_m - 1)/2 + 1) * 1e-3
    z_axis_m = 0

    # Get grid of measurement points
    z_m, x_m, y_m = np.meshgrid(z_axis_m, x_axis_m, y_axis_m)
    xyz_m = np.hstack((x_m, y_m, z_m))
    xyz_m = np.reshape(np.transpose(xyz_m, (0, 2, 1)), (-1, 3), order='F')
    n_measurement, _ = xyz_m.shape

    plt.figure(figsize=(8,6))
    plt.scatter(xyz_m[:,0],xyz_m[:,1],marker = 's', c= 'red')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")    
    plt.grid(True)
    plt.show()

    # Define target locations (coordinates: [x y z], x-Horizontal, y-Vertical, z-Depth)
    z_target_mm = self.target_distance

    # Apply multistatic to monostatic phase correction
    #----------------------------------------------------------
    # Calculate positions of TX and RX antennas [x,y,z] in m
    tx_ant_pos = np.zeros((n_channel, 3))
    rx_ant_pos = np.zeros((n_channel, 3))
    for i in range(n_channel):
      idx_tx_rx = self.vx_azimuth_selected[i][1]
      tx_ant_pos[i, :] = np.append(self.tx_pos_abs[idx_tx_rx[0]], 0.0)
      rx_ant_pos[i, :] = np.append(self.rx_pos_abs[idx_tx_rx[1]], 0.0)

    #TODO: Rotate board counter-clockwise
    for i in range(n_channel):
      tx_ant_pos[i, 0:2] = np.flip(tx_ant_pos[i, 0:2])
      rx_ant_pos[i, 0:2] = np.flip(rx_ant_pos[i, 0:2])

    y_tx_m = (np.max(tx_ant_pos[:, 1]) +  np.min(tx_ant_pos[:, 1]))/2
    y_rx_m = (np.max(rx_ant_pos[:, 1]) +  np.min(rx_ant_pos[:, 1]))/2
    for i in range(n_channel):
       tx_ant_pos[i, 1] = -(tx_ant_pos[i, 1] - y_tx_m) + y_tx_m
       rx_ant_pos[i, 1] = -(rx_ant_pos[i, 1] - y_rx_m) + y_rx_m

    #calculate virual antenna position [x,y,z] in m
    virtual_ch_pos = (tx_ant_pos + rx_ant_pos)/2
    virtual_ch_pos = np.reshape(virtual_ch_pos, (-1, 3))

    # Adapt positions of virtual channels because of TDM schedule.
    for i_chirp, item in enumerate(self.chirp_group_txvx_azimuth.items()):
       virtual_ch_pos[item[1], 0] = virtual_ch_pos[item[1], 0] 
       - i_chirp*self.t_chirp*self.platform_speed

    # set reference target position in center of virtual array or off-center
    # Target Off-center in x in m
    x_offset_target = 0
    x_target = (np.max(virtual_ch_pos[:,0])+np.min(virtual_ch_pos[:,0]))/2+ x_offset_target
    # Target Off-center in y in m
    y_offset_target = 0
    y_target = (np.max(virtual_ch_pos[:,1])+np.min(virtual_ch_pos[:,1]))/2+y_offset_target
    xyz_t = np.array([x_target,y_target,z_target_mm])             

    # plot all antennas and target position to check if calculation is true
    plt.figure(figsize=(8,6))
    plt.scatter(tx_ant_pos[:,0],tx_ant_pos[:,1],marker='s',c = "red")
    plt.scatter(rx_ant_pos[:,0],rx_ant_pos[:,1],marker = '^',c = "blue")
    plt.scatter(virtual_ch_pos[:,0],virtual_ch_pos[:,1],marker = 'o', c= 'green')
    plt.scatter(xyz_t[0],xyz_t[1], marker = 'x', c='black')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")    
    plt.grid(True)
    plt.show()

    tx_ant_pos = np.reshape(tx_ant_pos, (n_channel, 1, 3), order='F')
    rx_ant_pos = np.reshape(rx_ant_pos, (n_channel, 1, 3), order='F')

    # Calculate transmit and receive measurement positions in m
    xyz_m = np.reshape(xyz_m, (1, n_measurement, 3), order='F')
    xyz_m = np.tile(xyz_m, (n_channel, 1, 1))

    xyz_m_tx = xyz_m + tx_ant_pos
    xyz_m_rx = xyz_m + rx_ant_pos
    xyz_m_tx = np.reshape(xyz_m_tx, (-1, 3), order='F')
    xyz_m_rx = np.reshape(xyz_m_rx, (-1, 3), order='F')

    # Calculate monostatic transceiver measurement positions
    virtual_ch_pos = np.reshape(virtual_ch_pos, (n_channel, 1, 3), order='F')
    xyz_m_trx = xyz_m + virtual_ch_pos
    xyz_m_trx = np.reshape(xyz_m_trx, (-1, 3), order='F')

    # show all transmitter, receiver and transceiver elements in SAR-Apertur
    plt.figure(figsize=(8,6))
    plt.scatter(xyz_m_tx[:,0],xyz_m_tx[:,1],marker = 's', c= 'red')
    plt.scatter(xyz_m_rx[:,0], xyz_m_rx[:,1], marker = '^', c= 'blue')
    plt.scatter(xyz_m_trx[:,0], xyz_m_trx[:,1], marker = '^', c= 'green')
    plt.xlabel("X-Achse")
    plt.ylabel("Y-Achse")    
    plt.grid(True)
    plt.show()

    # Calculate distance matrix for multistatic
    # Distanz transmitter antennas to target
    r_tx_t = np.sqrt(np.sum((xyz_m_tx - xyz_t)**2, 1))
    
    # Distanz receiver antennas to target
    r_rx_t = np.sqrt(np.sum((xyz_m_rx - xyz_t)**2, 1))

    # Calculate distance matrix for monostatic -%
    r_trx_t = 2 * np.sqrt(np.sum((xyz_m_trx - xyz_t)**2, 1))

    # Signal reference multistatic
    k = np.transpose(np.squeeze(k))
    signal_ref_multistatic = np.exp(1j*(r_tx_t + r_rx_t)*k.T[:, np.newaxis])
    signal_ref_multistatic = np.reshape(signal_ref_multistatic.T, (n_channel, x_point_m, y_point_m, n_sample), order='F')
    signal_ref_multistatic = np.transpose(signal_ref_multistatic, axes=(0, 2, 1, 3))

    # Signal reference monostatic
    signal_ref_monostatic = np.exp(1j*r_trx_t*k.T[:, np.newaxis])
    signal_ref_monostatic = np.reshape(signal_ref_monostatic.T, (n_channel, x_point_m, y_point_m, n_sample), order='F')
    signal_ref_monostatic = np.transpose(signal_ref_monostatic, axes=(0, 2, 1, 3))

    sar_data_monostatic = sar_data_multistatic * signal_ref_monostatic / signal_ref_multistatic
    
    return sar_data_monostatic
  
  def reconstruct_sar_image(self, sar_data, frequency, x_step_mm, y_step_mm,z_target_mm, n_fft_kxy):
    """Reconstruct 3d image.

    Args:
      sar_data (numpy array): 
        SAR measurement data (format: yPointM x xPointM x nSample)
      frequency (numpy array): 
        Frequency data ([f_start, slope, f_sample, adc_start])
      x_step_mm (int): 
        Measurement step size in x-direction (horizontal) axis / mm
      y_step_mm (int): 
        Measurement step size in y-direction (vertical) axis / mm
      xy_size_t (int): 
        Size of target area (mm)
      z_target (int): 
       Target distance (mm)
      n_fft_kxy (list):
        Number of FFT points. Should be greater than x_step_mm and y_step_mm
    Returns:
      sarImage (numpy array):
        array containing reconstructed data for 2D Image
    """

    # Define frequency spectrum
    _, _, n_sample = sar_data.shape
    if (len(frequency) > 1) and (len(frequency) <= 4) and (n_sample > 1):
        f0, K, fS, adcStart = frequency
        f0 = f0 + adcStart*K; # This is for ADC sampling offset
        f = f0 + np.arange(0, n_sample)*K/fS
    else:
        print('Please correct the configuration and data for 3D processing.')

    # Define additional fixed parameters
    k = 2*np.pi*f/self.C
    k = np.reshape(k, (1, 1, -1), order='F')
    
    # Coincide aperture and target domains
    y_point_mm, x_point_mm, _ = sar_data.shape
    x_step_t = x_step_mm
    y_step_t = y_step_mm
    z_range_t_mm = z_target_mm

    # Define number of FFT points
    if (n_fft_kxy[0] < x_point_mm) or (n_fft_kxy[1] < y_point_mm):
        print('Number of FFT points should be greater than the number of measurement points. FFT will be performed at number of measurement points')
    if (n_fft_kxy[0] > x_point_mm):
        n_fft_kx = n_fft_kxy[0]
    else:
        n_fft_kx = x_point_mm
    if (n_fft_kxy[1] > y_point_mm):
        n_fft_ky = n_fft_kxy[1]    
    else:
        n_fft_ky = y_point_mm
    
    # Define wavenumbers using sampling frequency for target domain
    w_sx = 2*np.pi/(x_step_t * 1e-3)
    k_x = np.linspace(-(w_sx/2), (w_sx/2), n_fft_kx)
    
    # Define wavenumbers using sampling frequency for target domain
    w_sy = 2*np.pi/(y_step_t*1e-3)
    k_y = np.transpose(np.linspace(-(w_sy/2),(w_sy/2),n_fft_ky))

    # Zero padding to sar_data to locate target at center
    # Prepare the zero padded data
    sar_data_padded = np.zeros((n_fft_ky, n_fft_kx, n_sample), dtype='complex64')

    # Prepare the index of the data portion in x and y
    indexZeroPad_x = np.arange(np.floor((n_fft_kx - x_point_mm)/2), 
                                np.floor((n_fft_kx - x_point_mm)/2) + x_point_mm).astype(int)
    indexZeroPad_y = np.arange(np.floor((n_fft_ky - y_point_mm)/2), 
                                np.floor((n_fft_ky - y_point_mm)/2) + y_point_mm).astype(int)

    # Fill the zero padded data
    sar_data_padded[indexZeroPad_y[:, np.newaxis], indexZeroPad_x, :] = sar_data

    # Calculate k_z
    k_temp = np.squeeze(k)
    k_z = (2*k_temp[np.newaxis, np.newaxis, :])**2 - k_x[np.newaxis, :, np.newaxis]**2 - k_y[:, np.newaxis, np.newaxis]**2
    k_z = np.sqrt(k_z + 1j)



    # HACK
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(np.sum(np.abs(sar_data_padded),axis=-1))
    plt.show(block=False)



    # Take 2D FFT of SAR Data
    sar_data_fft = np.fft.fftshift(np.fft.fftshift(np.fft.fft2(sar_data_padded, axes=(0, 1)), 0), 1)

    # Create 2D-SAR Image for single z
    phase_factor = np.exp(-1j*z_range_t_mm*1e-3*k_z)
    k_temp = np.squeeze(k)
    idx_phase_factor = (k_x[np.newaxis, :, np.newaxis]**2 + k_y[:, np.newaxis, np.newaxis]**2) > (2*k_temp[np.newaxis, np.newaxis, :])**2
    phase_factor[idx_phase_factor] = 0

    sar_data_fft = k_z*sar_data_fft

    # HACK
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(np.sum(np.abs(sar_data_fft),axis=-1))
    plt.show(block=False)

    # HACK
    #phase_factor = np.ones_like(phase_factor)
    sar_data_fft = sar_data_fft*phase_factor


    # HACK
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(np.sum(np.abs(sar_data_fft),axis=-1))
    plt.show(block=False)


    sar_image = np.fft.ifft2(sar_data_fft, axes=(0, 1))
    sar_image = np.sum(sar_image, 2)

    # Define target axis
    x_range_t_mm = x_step_t * np.arange(-(n_fft_kx - 1)/2, (n_fft_kx - 1)/2+ 1)
    y_range_t_mm = y_step_t * np.arange(-(n_fft_ky - 1)/2, (n_fft_ky - 1)/2 + 1)

    # Plot the SAR image as a 2D mesh
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_range_t_mm, y_range_t_mm, np.abs(sar_image), cmap='gray', shading='auto')
    plt.colorbar()
    plt.xlabel('Horizontal / mm')
    plt.ylabel('Vertical / mm')
    plt.title(f"SAR image ({z_target_mm} mm)")
    plt.axis('equal')
    plt.grid(False)
    plt.show()
                                        
    return sar_image

  def sar_rda(self, s):
    """Apply Range Doppler Algorithm (RDA) SAR method for imaging.
    
    Args:
      s (numpy array):
        Signal data matrix s(x', y_R, y_T, k)

    Returns:

    Raises:
    """
    pass

  def plot_sar(self, p):
    """This function plots the reflectivity function.
    
    Args:
      p (numpy array): 
        Reflectivity tensor (x, y, z) (without units)

    Returns:

    Raises:
    """

    # Show SAR plot if desired
    if self.show_sar:

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')

      # Use the scatter function to plot
      x, y, z = np.indices(p.shape)
      scatter = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=p.flatten(), cmap='viridis')

      # Adding a color bar for reference
      fig.colorbar(scatter, ax=ax, label='Value')

      # Set labels
      ax.set_xlabel('X axis')
      ax.set_ylabel('Y axis')
      ax.set_zlabel('Z axis')
      ax.set_title('3D Scatter Plot of NumPy Array')

      # Show the plot
      plt.show(block=False)

  # endregion 

  # region Microdoppler

  def myspecgram(self, x, n, Fs, window, overlap):
    """Calculate spectrogram.
      
    Args:
      x (numpy array):
        Signal to be transformed
      n (int):
        FFT size
      Fs (float):
        Sampling frequency
      window (int): 
        Window size
      overlap (int):
        Overlap size
  
    Returns:
      f:                              TODO:
      t:                              TODO:
      stft (numpy array): 
        STFT

    Raises:
    """
      
    # Determine hanning window and parameters
    window_hann = self.hanning_window(window)
    win_size = np.size(window_hann, axis=1)
    if (win_size > n):
        n = win_size
        print("myspecgram(): FFT-size adjusted. Must be at least as long as frame.")
    step = win_size - overlap
    
    # Build matrix of windowed data slices
    offset = np.arange(1, len(x) - win_size,step)
    S = np.zeros((n,len(offset)),dtype='complex_')
    for i in range(len(offset)):
        S[0:win_size,i] = x[offset[i]-1:offset[i]+win_size-1]*window_hann
        
    # Compute FFT
    stft = sp.fft.fft(S,axis=0)
    
    # Extract the positive frequency components
    if np.remainder(n,2) == 1:
        ret_n = (n + 1)/2
    else:
        ret_n = n/2
        
    f = np.arange(0,ret_n)*Fs/n
    t = offset/Fs
    
    return f, t, stft

  def stft_basic(self, x, w, H=8, only_positive_frequencies=False):
    """Compute a basic version of the discrete short-time Fourier transform (STFT).
    
    Args:
      x (numpy array):
        Signal to be transformed
      w (numpy array):
        Window function
      H (int):
        Hop size (default: 8)
      only_positive_frequencies (bool):
        Return only positive frequency part of spectrum (non-invertible) (default: False)
  
    Returns:
      X (numpy array): 
        The discrete short-time Fourier transform

    Raises:
    """
    
    N = len(w)
    L = len(x)
    M = np.floor((L - N) / H).astype(int) + 1
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        x_win = x[m * H:m * H + N] * w
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
    
    return X

  # endregion

  # region Data import

  def read_data_ti_dca_1000(self, file_path):
      """Read binary data file captured by TI DCA1000 module.
      
      Read binary data file captured by TI DCA1000 module using mmWave Studio for IWR1443BOOST-Board.The data is usually arranged as follows when all four receivers are active:

      Start of file

      LVDS Lane 1: Frame 1 - Chirp 1 - Sample 1 - RX0 (I)
      LVDS Lane 2: Frame 1 - Chirp 1 - Sample 1 - RX1 (I)
      LVDS Lane 3: Frame 1 - Chirp 1 - Sample 1 - RX2 (I)
      LVDS Lane 4: Frame 1 - Chirp 1 - Sample 1 - RX3 (I)
      LVDS Lane 1: Frame 1 - Chirp 1 - Sample 1 - RX0 (Q)
      LVDS Lane 2: Frame 1 - Chirp 1 - Sample 1 - RX1 (Q)
      LVDS Lane 3: Frame 1 - Chirp 1 - Sample 1 - RX2 (Q)
      LVDS Lane 4: Frame 1 - Chirp 1 - Sample 1 - RX3 (Q)

      LVDS Lane 1: Frame 1 - Chirp 1 - Sample 2 - RX0 (I)
      LVDS Lane 2: Frame 1 - Chirp 1 - Sample 2 - RX1 (I)
      LVDS Lane 3: Frame 1 - Chirp 1 - Sample 2 - RX2 (I)
      LVDS Lane 4: Frame 1 - Chirp 1 - Sample 2 - RX3 (I)
      LVDS Lane 1: Frame 1 - Chirp 1 - Sample 2 - RX0 (Q)
      LVDS Lane 2: Frame 1 - Chirp 1 - Sample 2 - RX1 (Q)
      LVDS Lane 3: Frame 1 - Chirp 1 - Sample 2 - RX2 (Q)
      LVDS Lane 4: Frame 1 - Chirp 1 - Sample 2 - RX3 (Q)

      ...

      LVDS Lane 1: Frame 1 - Chirp 1 - Sample N - RX0 (I)
      LVDS Lane 2: Frame 1 - Chirp 1 - Sample N - RX1 (I)
      LVDS Lane 3: Frame 1 - Chirp 1 - Sample N - RX2 (I)
      LVDS Lane 4: Frame 1 - Chirp 1 - Sample N - RX3 (I)
      LVDS Lane 1: Frame 1 - Chirp 1 - Sample N - RX0 (Q)
      LVDS Lane 2: Frame 1 - Chirp 1 - Sample N - RX1 (Q)
      LVDS Lane 3: Frame 1 - Chirp 1 - Sample N - RX2 (Q)
      LVDS Lane 4: Frame 1 - Chirp 1 - Sample N - RX3 (Q)

      LVDS Lane 1: Frame 1 - Chirp 2 - Sample 1 - RX0 (I)
      LVDS Lane 2: Frame 1 - Chirp 2 - Sample 1 - RX1 (I)
      LVDS Lane 3: Frame 1 - Chirp 2 - Sample 1 - RX2 (I)
      LVDS Lane 4: Frame 1 - Chirp 2 - Sample 1 - RX3 (I)
      LVDS Lane 1: Frame 1 - Chirp 2 - Sample 1 - RX0 (Q)
      LVDS Lane 2: Frame 1 - Chirp 2 - Sample 1 - RX1 (Q)
      LVDS Lane 3: Frame 1 - Chirp 2 - Sample 1 - RX2 (Q)
      LVDS Lane 4: Frame 1 - Chirp 2 - Sample 1 - RX3 (Q)

      ...

      LVDS Lane 1: Frame 1 - Chirp M - Sample N - RX0 (I)
      LVDS Lane 2: Frame 1 - Chirp M - Sample N - RX1 (I)
      LVDS Lane 3: Frame 1 - Chirp M - Sample N - RX2 (I)
      LVDS Lane 4: Frame 1 - Chirp M - Sample N - RX3 (I)
      LVDS Lane 1: Frame 1 - Chirp M - Sample N - RX0 (Q)
      LVDS Lane 2: Frame 1 - Chirp M - Sample N - RX1 (Q)
      LVDS Lane 3: Frame 1 - Chirp M - Sample N - RX2 (Q)
      LVDS Lane 4: Frame 1 - Chirp M - Sample N - RX3 (Q)

      -> And consecutive for all frames!

      After arranging it the following tabular structure

      | RX0 (I,Q) | RX1 (I,Q) | RX2 (I,Q) | RX3 (I,Q) | 
      -------------------------------------------------
      Frame 1 - Chirp 1 - Sample 1 |           |           |           |           |
      ------------------------------------------------
      Frame 1 - Chirp 1 - Sample 2 |           |           |           |           |
      ...                          ------------------------------------------------
      Frame 1 - Chirp 1 - Sample N |           |           |           |           |
      ...                          ------------------------------------------------
      Frame 1 - Chirp M - Sample 1 |           |           |           |           |
      ------------------------------------------------
      Frame 1 - Chirp M - Sample 2 |           |           |           |           |
      ...                          ------------------------------------------------
      Frame 1 - Chirp M - Sample N |           |           |           |           |
      ...                          ------------------------------------------------
      Frame P - Chirp N - Sample 1 |           |           |           |           |
      ------------------------------------------------
      Frame P - Chirp N - Sample 2 |           |           |           |           |
      ...                          ------------------------------------------------
      Frame P - Chirp N - Sample N |           |           |           |           |
      ------------------------------------------------

      Args:
        file_path (str):
          File path

      Returns:
        adc_data (numpy array):
          ADC data

      Raises:
      """

      # Read binary data file. DCA1000 should read in two's complement data
      with open(file_path, 'rb') as fid:
          adc_data = np.fromfile(fid, dtype=np.int16)

      # If 12 or 14 bits ADC per sample compensate for sign extension
      if self.n_adc_bits != 16:
          l_max = 2**(self.n_adc_bits - 1) - 1
          mask = adc_data > l_max
          adc_data[mask] -= 2**self.n_adc_bits

      # Organize data by LVDS lane based on real or complex values
      if self.data_real:
          adc_data = np.reshape(adc_data, (self.n_lanes, -1), order='F')
      else:
          # Reshape to IWR1443+DCA1000-typical data structure 
          # (columns: first lanes -> real; last lanes -> complex values, rows: samples, then chirps)
          adc_data = np.reshape(adc_data, (self.n_lanes*2, -1), order='F')
          adc_data = adc_data[range(0, 4), :] + np.sqrt(-1)*adc_data[range(4, 8), :]

      # Plot information about data as interpreted by the parameters
      if self.plot_information:

        # Plot general information
        print(f'Data read.')
        print(f'Shape: {adc_data.shape[0]} x {adc_data.shape[1]}')
        print(f'Number of lanes: {adc_data.shape[1]}')
        print(f'Number of samples per frame: {adc_data.shape[1]/self.n_frames}')
        print(f'Number of samples per chirp: {adc_data.shape[1]/self.n_frames/self.n_chirps}')
        if int(adc_data.shape[1]/self.n_frames/self.n_chirps) == self.n_samples:
            print(f'Data dimensions seem to be consistent.')

      return adc_data
  
  def print_parameter_file(self, parameters_json):
    """Print important entries of the JSON file object of the measurement.
      
    Args:
      parameters_json (JSON object):
        Parameters

    Returns:

    Raises:
    """

    # Show parameters of parameter file if desired
    if self.show_parameter_file:

      print('-------------------------------------------')
      print('read_data_ti_mmwcas_dsp(): Measurement data')
      print('-------------------------------------------')
      print('Created by: ' + parameters_json['configGenerator']['createdBy'])
      print('Created on: ' + parameters_json['configGenerator']['createdOn'])
      print('mmWavLink Version: ' + str(parameters_json['currentVersion']['mmwavelinkVersion']['major']) + '.' + str(parameters_json['currentVersion']['mmwavelinkVersion']['minor']) + '.' + str(parameters_json['currentVersion']['mmwavelinkVersion']['patch']))
      print('SDK Version: ' + str(parameters_json['currentVersion']['SDKVersion']['major']) + '.' + str(parameters_json['currentVersion']['SDKVersion']['minor']) + '.' + str(parameters_json['currentVersion']['SDKVersion']['patch']))
      print('DFP Version: ' + str(parameters_json['currentVersion']['DFPVersion']['major']) + '.' + str(parameters_json['currentVersion']['DFPVersion']['minor']) + '.' + str(parameters_json['currentVersion']['DFPVersion']['patch']))
      print('JSON Config. Version: ' + str(parameters_json['currentVersion']['jsonCfgVersion']['major']) + '.' + str(parameters_json['currentVersion']['jsonCfgVersion']['minor']) + '.' + str(parameters_json['currentVersion']['jsonCfgVersion']['patch']))
      print('- - - - - - - - ')
      print('Scene parameters')
      print('- - - - - - - - ')
      print('Ambient temperature: ' + str(parameters_json['systemConfig']['sceneParameters']['ambientTemperature_degC']))
      print('Max. detectable range (m): ' + str(parameters_json['systemConfig']['sceneParameters']['maxDetectableRange_m']))
      print('Range resolution (cm): ' + str(parameters_json['systemConfig']['sceneParameters']['rangeResolution_cm']))
      print('Max. velocity (km/h): ' + str(parameters_json['systemConfig']['sceneParameters']['maxVelocity_kmph']))
      print('Velocity Resolution (km/h): ' + str(parameters_json['systemConfig']['sceneParameters']['velocityResolution_kmph']))
      print('Measurement rate: ' + str(parameters_json['systemConfig']['sceneParameters']['measurementRate']))
      print('Typical detected Object RCS: ' + str(parameters_json['systemConfig']['sceneParameters']['typicalDetectedObjectRCS']))
      print('Number of devices: ' + str(len(parameters_json['mmWaveDevices'])))
      for i in range(len(parameters_json['mmWaveDevices'])):
        print('- - - - ')
        print('Device' + str(i))
        print('- - - - ')
        print('RF config / Waveform type | MIMO-Scheme: ' + parameters_json['mmWaveDevices'][i]['rfConfig']['waveformType'] + ' | ' + parameters_json['mmWaveDevices'][i]['rfConfig']['MIMOScheme'] )
        
        print('Frame config. / Chirp start index | Chirp end index | Number of loops (blocks) | Number of frames | Frame periodicity (msec) | Trigger select | Frame Trigger Delay: ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['chirpStartIdx']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['chirpEndIdx']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['numLoops']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['numFrames']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['framePeriodicity_msec']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['triggerSelect']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['frameTriggerDelay']))

        print('Number of profiles: ' + str(len(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'])))
        for j in range(len(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'])):
            print('Chirp profile ' + str(j) + ' / Start freq. (GHz) | Idle Time (usec) | TX start time (usec) | ADC start time (usec) | Ramp end time (usec) | Slope (MHz/usec) | Num. ADC samples: ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['startFreqConst_GHz']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['idleTimeConst_usec']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['txStartTime_usec']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['adcStartTimeConst_usec']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['rampEndTime_usec']) + ' | ' +  str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['freqSlopeConst_MHz_usec']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['numAdcSamples']))

        print('Number of chirps: ' + str(len(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'])))
        for j in range(len(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'])):
            print('Chirp ' + str(j) + ' / Chirp start index | Chirp end index | Profile ID | Start freq. variation (MHz) | Slope variation (KHz/usec) | Idle time variation (usec) | ADC start time variation (usec): ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'][j]['rlChirpCfg_t']['chirpStartIdx']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'][j]['rlChirpCfg_t']['chirpEndIdx']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'][j]['rlChirpCfg_t']['profileId']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'][j]['rlChirpCfg_t']['startFreqVar_MHz']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'][j]['rlChirpCfg_t']['freqSlopeVar_KHz_usec']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'][j]['rlChirpCfg_t']['idleTimeVar_usec']) + ' | ' + str(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps'][j]['rlChirpCfg_t']['adcStartTimeVar_usec']))

  def extract_parameter_file(self, parameters_json):
    """Extract parameters from parameter file.
    
    Extract parameters from parameter file. It is assumed that all devices are configured in TDM-MIMO mode based on one chirp profile and all device-related parameters are equal and that all chirp variations are zero.

    Args:
      parameters_json (JSON object):
        Parameters

    Returns:
      n_frames (int):
        Number of frames
      n_loops (int): 
        Number of loops (blocks)
      n_chirps (int):
        Number of chirps
      n_bits (int):
        Number of samples
      adc_format (int):
        ADC format (0: 'real'; 1: 'complex'; 2: 'complex with image band'; '3': 'pseudo-real')
      freq_start (float): 
        Start frequency in Hz
      t_idle (float): 
        Idle time in s
      t_tx_start (float): 
        TX start time in s
      t_adc_start (float): 
        ADC start time in s
      time_ramp_end (float): 
        Ramp time in s
      n_samples (int):
        Number of samples
      slope (float): 
        Slope in Hz/s
      sample_rate (float): 
        Sample rate in sps (samples/s)

    Raises:
      RuntimeError: 
        Parameters are not consistent.
    """

    n_frames = []
    n_loops = []
    n_chirps = []
    n_bits = []
    adc_format = []
    freq_start = []
    t_idle = []
    t_tx_start = []
    t_adc_start = []
    time_ramp_end = []
    n_samples = []
    slope = []
    sample_rate = []
    for i in range(len(parameters_json['mmWaveDevices'])):
      n_frames.append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['numFrames'])
      n_loops.append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlFrameCfg_t']['numLoops'])
      n_chirps.append(len(parameters_json['mmWaveDevices'][i]['rfConfig']['rlChirps']))
      n_bits.append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlAdcOutCfg_t']['fmt']['b2AdcBits'])
      adc_format.append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlAdcOutCfg_t']['fmt']['b2AdcOutFmt'])
      freq_start.append([])
      t_idle.append([])
      t_tx_start.append([])
      t_adc_start.append([])
      time_ramp_end.append([])
      n_samples.append([])
      slope.append([])
      sample_rate.append([])
      for j in range(len(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'])):
        freq_start[i].append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['startFreqConst_GHz'] * 1e9)
        t_idle[i].append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['idleTimeConst_usec'] * 1e-6)
        t_tx_start[i].append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['txStartTime_usec'] * 1e-6)
        t_adc_start[i].append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['adcStartTimeConst_usec'] * 1e-6)
        time_ramp_end[i].append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['rampEndTime_usec'] * 1e-6)
        n_samples[i].append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['numAdcSamples'])
        slope[i].append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['freqSlopeConst_MHz_usec'] * 1e12)
        sample_rate[i].append(parameters_json['mmWaveDevices'][i]['rfConfig']['rlProfiles'][j]['rlProfileCfg_t']['digOutSampleRate'] * 1e3)

    if not (all(x == n_frames[0] for x in n_frames) and \
            all(x == n_loops[0] for x in n_loops) and \
            all(x == n_chirps[0] for x in n_chirps) and \
            all(x == n_bits[0] for x in n_bits) and \
            all(x == adc_format[0] for x in adc_format) and \
            all(x == freq_start[0][0] for x in freq_start[0]) and \
            all(x == t_idle[0][0] for x in t_idle[0]) and \
            all(x == t_tx_start[0][0] for x in t_tx_start[0]) and \
            all(x == t_adc_start[0][0] for x in t_adc_start[0]) and \
            all(x == time_ramp_end[0][0] for x in time_ramp_end[0]) and \
            all(x == n_samples[0][0] for x in n_samples[0]) and \
            all(x == slope[0][0] for x in slope[0]) and \
            all(x == sample_rate[0][0] for x in sample_rate[0])):
      n_frames = []
      n_loops = []
      n_chirps = []
      n_bits = []
      adc_format = []
      freq_start = []
      t_idle = []
      t_tx_start = []
      t_adc_start = []
      time_ramp_end = []
      n_samples = []
      n_bits = []
      slope = []
      sample_rate = []
      raise RuntimeError('Parameters not consistent. Please check the parameter file.')
    
    # Transform ADC bits ID to ADC bit number
    if n_bits[0] == 0:
        n_bits = 12
    elif  n_bits[0] == 1:
        n_bits = 14
    elif  n_bits[0] == 2:
        n_bits = 16

    # Transform ID of ADC format to a description
    if adc_format[0] == 0:
        adc_format = 'real'
    elif adc_format[0] == 1:
        adc_format = 'complex'
    elif adc_format[0] == 2:
        adc_format = 'complex with image band'
    elif adc_format[0] == 3:
        adc_format = 'pseudo-real'

    # Temp: Board issue. Not all frames are collected. Unknown reason!
    n_frames[0] = n_frames[0]

    return n_frames[0], n_loops[0], n_chirps[0], n_bits, adc_format, freq_start[0][0], t_idle[0][0], t_tx_start[0][0], t_adc_start[0][0], time_ramp_end[0][0], n_samples[0][0], slope[0][0], sample_rate[0][0]
        
  def read_bin_file(self, file_paths, n_rx_per_device, frame_idx, n_frames, n_blocks, n_chirps, n_samples, n_bits=16, dict_file_name_adc_idx_start=[]):
    """Read binary data file captured by one device in the TI MMWCAS-RF-EVM. 
    
    The board is attached to an TI MMWCAS-DSP-EVM for the full TDA mode (12 Tx, 16 Rx).

    Args:
      file_paths (str):
        File paths
      n_rx_per_device (int):
        Number of RX antennas
      frame_idx (int):
        Current frame index
      n_blocks (int):
        Number of blocks as integer
      n_chirps (int):
        Number of chirps per block
      n_samples (int):
        Number of samples
      n_bits (int):
        Number of bits per sample
      dict_file_name_adc_idx_start (dict):
        Name-index pair (default: None)

    Returns:
      adc_data_complex_agg (numpy array):
        Number of ADC data

    Raises:
    """

    # Determine expected number of samples per frame
    n_samples_per_frame = n_rx_per_device*self.n_samples*n_chirps*n_blocks*2

    # Open the file in binary read modes
    adc_data_complex_agg = np.zeros((n_samples, n_blocks, n_rx_per_device*4, n_chirps), dtype=np.complex128)
    for i in range(len(file_paths)):
      with open(file_paths[i], 'rb') as fp:

        # Proceed only if number of elements in current file match with given parameters
        n_elements_read = os.path.getsize(file_paths[i])
        n_elements_expected = int(n_frames*n_blocks*n_chirps*n_rx_per_device*n_samples*(n_bits/4))

        if n_elements_read == n_elements_expected:

          # Seek to the specified frame index by setting symbol cursor and read the ADC data (all samples of the current frame) as unsigned 16-bit integers #uint16->int16

          fp.seek(frame_idx*n_samples_per_frame*np.uint16(1).nbytes, 0)
          adc_data = np.fromfile(fp, dtype=np.int16, count=n_samples_per_frame) 

          # Combine the real and imaginary parts. Here it is assumed that the real and complex value are arranged alternately
          adc_data = adc_data[::2] + 1j * adc_data[1::2]

          # Reshape the data (n_samples x n_blocks x n_rx(per_chip) x n_chirps)
          # adc_data_reshaped = np.reshape(adc_data, [self.n_rx_chip, 
          #                                           self.n_samples, 
          #                                           n_chirps, 
          #                                           n_blocks], order='F') # F
          # adc_data_reshaped = np.transpose(adc_data_reshaped, (1, 3, 0, 2))
          # ORIGINAL!


          adc_data_reshaped = np.reshape(adc_data, [self.n_samples, 
                                                    self.n_rx_chip, 
                                                    n_chirps, 
                                                    n_blocks], order='F') # F
          adc_data_reshaped = np.transpose(adc_data_reshaped, (0, 3, 1, 2))  
          # MOST PLAUSIBLE!

          # adc_data_reshaped = np.reshape(adc_data, [self.n_samples, 
          #                                           n_chirps, 
          #                                           self.n_rx_chip, 
          #                                           n_blocks], order='F') # F
          # adc_data_reshaped = np.transpose(adc_data_reshaped, (0, 3, 2, 1))


          # adc_data_reshaped = np.reshape(adc_data, [self.n_samples, 
          #                                           n_blocks, 
          #                                           self.n_rx_chip, 
          #                                           n_chirps], order='F') # F
          # adc_data_reshaped = np.transpose(adc_data_reshaped, (0, 1, 2, 3))
          # TRASH!

          # adc_data_reshaped = np.reshape(adc_data, [self.n_samples, 
          #                                           n_chirps, 
          #                                           n_blocks,
          #                                           self.n_rx_chip], order='F') # F
          # adc_data_reshaped = np.transpose(adc_data_reshaped, (0, 2, 3, 1))
          # TRASH!

          # adc_data_reshaped = np.reshape(adc_data, [self.n_samples, 
          #                                           n_blocks, 
          #                                           n_chirps,
          #                                           self.n_rx_chip], order='F') # F
          # adc_data_reshaped = np.transpose(adc_data_reshaped, (0, 1, 3, 2))
          # TRASH!

          # adc_data_reshaped = np.reshape(adc_data, [self.n_samples, 
          #                                           self.n_rx_chip, 
          #                                           n_blocks,
          #                                           n_chirps], order='F') # F
          # adc_data_reshaped = np.transpose(adc_data_reshaped, (0, 2, 1, 3))
          # TRASH!

          # TEMP
          y_lim_up = 1000
          y_lim_low = -1000

          # _, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
          # ax[0,0].plot(np.imag(adc_data_reshaped[:, 0, 0, 0]), color='red', label='imag')
          # ax[0,0].plot(np.real(adc_data_reshaped[:, 0, 0, 0]), color='blue', label='real')
          # ax[0,0].legend()
          #ax[0,0].set_ylim(y_lim_low, y_lim_up)

          # ax[0,1].plot(np.imag(adc_data_reshaped[:, 0, 1, 0]), color='red', label='imag')
          # ax[0,1].plot(np.real(adc_data_reshaped[:, 0, 1, 0]), color='blue', label='real')
          # ax[0,1].legend()
          # ax[0,1].set_ylim(y_lim_low, y_lim_up)

          # ax[1,0].plot(np.imag(adc_data_reshaped[:, 0, 2, 0]), color='red', label='imag')
          # ax[1,0].plot(np.real(adc_data_reshaped[:, 0, 2, 0]), color='blue', label='real')
          # ax[1,0].legend()
          # ax[1,0].set_ylim(y_lim_low, y_lim_up)

          # ax[1,1].plot(np.imag(adc_data_reshaped[:, 0, 3, 0]), color='red', label='imag')
          # ax[1,1].plot(np.real(adc_data_reshaped[:, 0, 3, 0]), color='blue', label='real')
          # ax[1,1].legend()
          # ax[1,1].set_ylim(y_lim_low, y_lim_up)

          plt.show(block=False)

          file_paths_split = file_paths[i].split("\\")[-1][:6]
          idx_start = dict_file_name_adc_idx_start[file_paths_split]
          idx_end = dict_file_name_adc_idx_start[file_paths_split] + n_rx_per_device
          adc_data_complex_agg[:, :, idx_start:idx_end, :] = adc_data_reshaped

        else:
            raise ValueError(f"Number of elements in current doesnt match with the parameters. The expected number of elements for the set (n_bits: {n_bits}, n_samples: {n_samples}, n_blocks: {n_blocks}, n_chirps: {n_chirps}, n_frames: {n_frames}) is {n_elements_expected}, but {n_elements_read} were found.")
    
    return adc_data_complex_agg

  def read_data_ti_mmwcas_dsp(self, orientation, n_devices, n_tx_all, n_rx_per_device, adc_calibration_on=True, phase_calibration_only=False, dict_file_name_adc_idx_start=[], output_format=[]):       
    """Read binary data file captured by TI MMWCAS-RF-EVM.
    
    Read binary data file captured by TI MMWCAS-RF-EVM that is attached to an TI MMWCAS-DSP-EVM.

    Args:
      n_devices (int):
        Number of devices
      n_tx_all (int):
        Number of all TX antennas
      n_rx_per_device (int):
        Number of RX antennas per device
      n_frames (int):
        Number of frames
      n_blocks (int):
        Number of blocks
      n_chirps (int):
        Number of chirps
      n_samples (int):
        Number of samples
      n_bits (int):
        Number of bits
      sampling_rate (int):
        Sampling rate in ksps
      slope (float):
        Slope in Hz/s
      adc_calibration_on (bool):
        Flag for ADC calibration
      phase_calibration_only (bool):
        Flag for normalization
      dict_file_name_adc_idx_start (dict): 
        Devices and corresponding ADC indices
      output_format (str):
        Output format

    Returns:
      out_data_final (numpy array):
        Output data

    Raises:
    """
    
    # Get the current directory, change to specified folder, find the binary data files and change back
    data_files = [f for f in os.listdir(self.measurement_folder_path) 
                  if f.endswith("_data.bin")]
    idx_files = [f for f in os.listdir(self.measurement_folder_path) 
                 if f.endswith("_idx.bin")]

    # Check for completeness of files
    if len(data_files) != 4 or len(idx_files) != 4:
      raise FileExistsError('Files incomplete.')
    else:
      # Get file paths
      paths_data_files = []
      paths_idx_files = []
      for i in range(len(data_files)):
        paths_data_files.append(os.path.join(self.measurement_folder_path, data_files[i]))
        paths_idx_files.append(os.path.join(self.measurement_folder_path, idx_files[i]))

      # Since TI's radar board misses frames in specific cases, check if the number of expected bytes match the number of found bytes for all devices
      n_elements_read = [0 for i in range(4)]
      n_bits = 16
      n_elements_expected = int(self.n_frames*self.n_blocks*self.n_chirps*n_rx_per_device*self.n_samples*(n_bits/4))
      for i in range(4):
        n_elements_read[i] = os.path.getsize(paths_data_files[i])
      if all(x == n_elements_read[0] for x in n_elements_read):
        if n_elements_read[0] != n_elements_expected:
          self.n_frames = int(n_elements_read[0]/(self.n_blocks*self.n_chirps*self.n_samples*self.n_adc_bits))
          print(f"Number of frames has to be adjusted prior tho data reading since the designated number and the actual number do not match (Number expected: {n_elements_expected}; Number read: {n_elements_read[0]})")

      # Apply data conversion for all devices for every frame
      adc_data_complex_all = []
      for h in range(self.n_frames):
          adc_data_complex_all.append(self.read_bin_file(paths_data_files,  n_rx_per_device, h, self.n_frames, self.n_blocks, self.n_chirps, self.n_samples, self.n_adc_bits, dict_file_name_adc_idx_start))

      # Filter data
      # ub_val_abs = 2.0e3
      # for i, adc_data_complex in enumerate(adc_data_complex_all):
      #    adc_data_complex[np.abs(adc_data_complex) > ub_val_abs] = 0
      #    adc_data_complex_all[i] = adc_data_complex

      # Calibrate if desired
      if not adc_calibration_on:
        # Format data
        out_data_final = self.format_output(adc_data_complex_all, 
                                            orientation, 
                                            self.n_frames, 
                                            output_format)
      else:
        # Calibrate and format data
        calibrated_data = self.calibrate_data(adc_data_complex_all, 
                                              self.parameter_file_path, 
                                              self.calibration_file_path, 
                                              n_devices, n_rx_per_device, n_tx_all, self.n_frames, 
                                              self.n_blocks, self.n_chirps, 
                                              self.n_samples, self.sample_rate, self.sweep_slope, phase_calibration_only)
        out_data_final = self.format_output(calibrated_data, orientation, self.n_frames, output_format)

    # HACK
    #self. plot_raw_signals(adc_data_complex_all, i_frame=0, i_block=1)

    return out_data_final

  def plot_raw_signals(self, data, i_frame=0, i_block=1):
    '''Plot raw signals of every virtual channel (assuming 86 channels are given)
    
    Args:
      data (list): 
        List of frames. Every frame has a numpy matrix of size 
        n_s x n_b x n_TX x n_RX.
      i_frame (int):
        Index of frame to plot (default: 0)
      i_block (int):
        Index of block to plot (default: 1)
    Returns:

    Raises:
    '''

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    fig.canvas.manager.set_window_title(f"Frame: {i_frame} / Block: {i_block}")

    for i in range(10):
      ax[0,0].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[0,0].set(xlabel="sample", ylabel="value", title="VX 1-10")
    ax[0,0].legend()

    for i in range(10, 20):
      ax[0,1].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[0,1].set(xlabel="sample", ylabel="value", title="VX 11-20")
    ax[0,1].legend()

    for i in range(20, 30):
      ax[1,0].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[1,0].set(xlabel="sample", ylabel="value", title="VX 21-30")
    ax[1,0].legend()

    for i in range(30, 40):
      ax[1,1].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[1,1].set(xlabel="sample", ylabel="value", title="VX 31-40")
    ax[1,1].legend()


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    fig.canvas.manager.set_window_title(f"Frame: {i_frame} / Block: {i_block}")

    for i in range(40, 50):
      ax[0,0].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[0,0].set(xlabel="sample", ylabel="value", title="VX 41-50")
    ax[0,0].legend()

    for i in range(50, 60):
      ax[0,1].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[0,1].set(xlabel="sample", ylabel="value", title="VX 51-60")
    ax[0,1].legend()

    for i in range(60, 70):
      ax[1,0].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[1,0].set(xlabel="sample", ylabel="value", title="VX 61-70")
    ax[1,0].legend()

    for i in range(70, 80):
      ax[1,1].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[1,1].set(xlabel="sample", ylabel="value", title="VX 71-80")
    ax[1,1].legend()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    fig.canvas.manager.set_window_title(f"Frame: {i_frame} / Block: {i_block}")

    for i in range(80, 86):
      ax[0,0].plot(data[i_frame][:, i_block, int(self.vx_azimuth_selected[i][1][1])-1, int(self.vx_azimuth_selected[i][1][0])-1].T, label=f"VX{i+1}")
    ax[0,0].set(xlabel="sample", ylabel="value", title="VX 81-86")
    ax[0,0].legend()

    plt.show(block=False)
    pass


  def calibrate_data(self, adc_data_complex_all, json_file_path, calibration_file_path, n_devices, n_rx_per_device, n_tx_all, n_frames, n_blocks, n_chirps, n_samples, sampling_rate, slope, phase_calibration_only):
    """Calibrate data.

    For this, calibration files that have been generated before are required.
    
    Args:
      adc_data_complex_all (numpy array):
        ADC data for calibration
      json_file_path (str):
        JSON file path
      calibration_file_path (str): 
        Calibration file path
      n_devices (int):
        Number of devices
      n_rx_per_device (int):
        Number of RX antennas per device
      n_tx_all (int):
        Number of all TX antennas
      n_frames (int):
        Number of frames
      n_blocks (int):
        Number of blocks
      n_chirps (int): 
        Number of chirps
      n_samples (int):
        Number of samples
      sampling_rate (float):
        Sampling rate in ksps
      slope (float):
        Slope in Hz/s
      phase_calibration_only (bool):
        Flag for normalization

    Returns:
      out_data_final (numpy array): 
      Calibrated data

    Raises:
    """

    # Initialize radar cube array
    calibrated_data = [np.zeros((n_samples, n_blocks, n_rx_per_device*n_devices, n_chirps), dtype=np.complex128) for i in range(n_frames)]

    # Get calibration matrices
    calibration_data = sp.io.loadmat(calibration_file_path)
    calibration_results = calibration_data['calibResult']
    range_mat_calib = calibration_results['RangeMat'][0][0]     # R^(12 x 16)
    peaks_mat_calib = calibration_results['PeakValMat'][0][0]   # R^(12 x 16)
    
    # Display data in JSON file
    with open(json_file_path, "r") as file:
        config = json.load(file)
        if self.show_calibration_parameters:
          print('---------------------------')
          print('Calibrate data / parameters')
          print('---------------------------')
          print(f"JSON file ({json_file_path}):")
          print(print(json.dumps(config, indent=None)))

    # Get parameters
    calibration_parms = calibration_data['params']
    params_names = list(calibration_parms.dtype.fields.keys())
    for i in range(len(params_names)):
      if params_names[i] == 'Slope_MHzperus':
        slope_calib = calibration_parms[0][0][i][0][0]*1e12            # Slope of chirp in MHz/(1e-6 s) / Example: 7.8986e+13 Hz/s -> 7.8986e+13 Hz/s
      elif params_names[i] == 'Sampling_Rate_sps':
        sampling_rate_sps = calibration_parms[0][0][i][0][0]           # Sampling rate / Example: 8e6 sps
    calibration_interp = 5                                             # Example: 5      
    # slope_calib = config['Slope_calib']
    # sampling_rate_sps = config['Sampling_Rate_sps']
    # calibration_interp = config['Slope_calib']
    # fs_calib = config['fs_calib']
        
    # Set TX order of TDM schedule and use the first TX as reference by default
    tx_to_enable = np.flip(np.arange(0, 12, 1))
    tx_ref = tx_to_enable[0]

    # Apply for all frames
    for h in range(n_frames):
      # Apply for all TX antennas
      for i_tx in range(n_tx_all):
          
          # Use first enabled TX1/RX1 as reference for calibration
          tx_idx = tx_to_enable[i_tx]

          # Construct the frequency compensation matrix using the index differences of the range peaks.
          peak_index_diff = (range_mat_calib[tx_idx, :] - 
                             range_mat_calib[tx_ref, 0])
          freq_calib = 2 * np.pi * peak_index_diff * \
                       sampling_rate_sps / sampling_rate *\
                       slope / slope_calib / (n_samples * calibration_interp) # (R^(1 x n_rx_all))
          correction_vec = np.transpose (np.exp(1j * (np.arange(n_samples)[:, np.newaxis] * freq_calib))) # C^(n_rx_all x n_s)
          freq_correction_mat = np.transpose(np.repeat(correction_vec[:, :, np.newaxis], n_blocks, axis=-1), (1, 2, 0)) # C^(n_s x n_b x n_rx_all)

          # Multiplication leads finally to frequency mismatch compensation
          out_data_1_tx = adc_data_complex_all[h][:, :, :, i_tx] * \
          freq_correction_mat # C^(n_s x n_b x n_rx_all)
          
          # Construct the phase compensation matrix in the next step
          phase_calib = peaks_mat_calib[tx_ref, 0] \
                        / peaks_mat_calib[tx_idx, :] # C^(1 x n_rx_all)
          if phase_calibration_only == 1:
              phase_calib = phase_calib / np.abs(phase_calib)          
          #phase_calib_T = np.transpose(phase_calib) # C^(1 x n_rx_all)
          phase_correction_mat = np.repeat(phase_calib[:, np.newaxis], n_samples, axis=-1) # C^(n_rx_all x n_s)
          phase_correction_mat = np.transpose(np.repeat(phase_correction_mat[:, :, np.newaxis], n_blocks, axis=-1), (1, 2, 0))  # C^(n_s x n_b x n_rx_all

          # Second multiplication leads to phase (if desired, amplitude, too) compensation
          calibrated_data[h][:, :, :, i_tx] = out_data_1_tx * phase_correction_mat # C^(n_s x n_b x n_rx_all)
          
    return calibrated_data

  def format_output(self, data, orientation, n_frames, output_format):
    """Format output based on desired output format.
    
    Args:
      data (numpy array): 4D numpy array containing data for formatting
      orientation (str): 
        Orientation ('azimuth': Azimuth; 'elevation': Elevation
      n_frames (int):
        Number of frames
      output_format (str):
        Desired output data as string ('3d-unsorted'; '4d-unsorted';'3d-sorted')

    Returns:
      data_formatted (numpy array):
        Formatted output data

    Raises:
    """

    if output_format =='3d-unsorted':

      # Swap both last dimensions and apply reshapping so that elements of the last dimension change faster than the penultimate one.
      for h in range(n_frames):
          data[h] = np.swapaxes(data[h], -1, -2)
          out_data_final = data.copy()
          out_data_final[h] = np.zeros((data[h].shape[0], data[h].shape[1], data[h].shape[2]*data[h].shape[3]))
          a, b, c, d = data[h].shape
          out_data_final[h] = data[h].reshape(a, b, c * d, order='C')

    elif output_format == '4d-unsorted':

      out_data_final = data

    elif output_format == '3d-sorted':

      out_data_final = [0.0 for i in range(n_frames)]
      for h in range(n_frames):
        if orientation == 'azimuth':
            data[h] = np.swapaxes(data[h], -1, -2)
            indices_tx = [(int(item[1][0])-1) 
                          for item in self.vx_azimuth_selected]
            indices_rx = [(int(item[1][1])-1) 
                          for item in self.vx_azimuth_selected]
            out_data_final[h] = data[h][:, :, indices_tx, indices_rx]
            pass

        elif orientation == 'elevation':
            data[h] = np.swapaxes(data[h], -1, -2)
            indices_tx = [(int(item[1][0])-1) 
                          for item in self.vx_elevation_selected]
            indices_rx = [(int(item[1][1])-1) 
                          for item in self.vx_elevation_selected]
            out_data_final[h] = data[h][:, :, indices_tx, indices_rx]
    else:
      raise TypeError('Output format unknown.')
      out_data_final = []

    return out_data_final

  # endregion

  # region Auxiliary functions

  def normalize(self, x):
    """Normalize data structure between 0 and 1.
    
    Args:
      x (numpy array; float):
        Data

    Returns:
      x_norm (numpy array; float):
        Normalized data

    Raises:
    """
    x_norm = (x - np.min(x))/(np.max(x) - np.min(x))
    return x_norm

  def oddnumber(self, x):
    """Calculate the nearest odd number for a given number.
    
    Args:
      x (int):
        Number

    Returns:
      y (int):
        Nearest odd number

    Raises:
    """
    
    # Get the nearest odd number of x
    if type(x) is list:
        nx = len(x)
        y = np.zeros((1,nx))
        for k in range(1,nx+1):
            y[k] = np.floor(x[k])
            if np.remainder(y[k],2) == 0:
                y[k] = np.ceil(x[k])
            if np.remainder(y[k],2) == 0:
                y[k] = y[k] + 1
    else:
        nx = 1
        y = np.floor(x)
        if np.remainder(y,2) == 0:
            y = np.ceil(x)
        if np.remainder(y,2) == 0:
            y = y + 1
                        
    return y
  
  # TODO: Remove if already provided.
  def hanning_window(self, N):
    """Create Hanning window.
    
    Create Hanning window, where the zero elements are removed. If this does the same like the function 'hanning_matlab, remove.
    
    Args:
      N (int):
        Size

    Returns:
      W (numpy array):
        Hanning window
    
    Raises:
    """       

    if N % 2 == 0:
        M = int(N/2)
    else:
        M = int((N + 1)/2)
        
    w_f = np.zeros((1,M-2))
    for n in range(1,M-1):
        w_f[0,n-1] = 0.5*(1.0 - np.cos(2*np.pi*n/N))
    w = np.hstack((w_f,np.fliplr(w_f)))
    
    return w

  # endregion 

  # region Main calculation

  def range_velocity_angle_computation(self, radar_data_cube_frames=None, orientation=None, frame_range=None, idx_frame_plot=0, idx_channel_rd_plot=0):
    """Apply signal processing techniques for the determination of range, velocity and angle of targets.
    
    Args:
      radar_data_cube_frames (list):
        Frames containing radar cube data in a specific direction (default: None). If none, data from measurement data folder (self.measurement_folder_path) are read.
      orientation (str):
        Orientation: 'azimuth': Azimuth; 'elevation': Elevation (default: None)           
      frame_range (list of integers):
        Frame range: [index_start, index_end] (default: None)
      idx_frame_plot (int):
        Index of frame (default: 0)
      idx_channel_plot (int):
        Index of channel for RD plot (default: 0)

    Returns:

    Raises:
    """

    # Read measurement data from specific board if no frames available
    if radar_data_cube_frames is None:
      if self.board_name == 'IWR6843_AOP_REV_G':
        raise NotImplementedError('Data import function for this board not implemented yet.')      # TODO
      elif self.board_name == 'IWR1443_BOOST':
        raise NotImplementedError('Data import function for this board not implemented yet.')      # TODO
      elif self.board_name == 'MMWCAS_RF_DSP':  
        radar_data_cube_frames = self.read_data_ti_mmwcas_dsp(orientation=orientation, 
        n_devices=4, n_tx_all=self.n_tx_all, n_rx_per_device=4,
        adc_calibration_on=True, phase_calibration_only=True,
        dict_file_name_adc_idx_start = {'master': 0, 'slave1': 4, 'slave2': 8, 'slave3': 12}, output_format='3d-sorted')

    # Preallocate
    if frame_range is not None:
      frame_start = frame_range[0]
      frame_end = frame_range[1]
    else:
      frame_start = 0
      frame_end = self.n_frames
    range_data = [None for i in range(frame_start, frame_end)]
    doppler_data = [None for i in range(frame_start, frame_end)]
    doppler_data_sum = [None for i in range(frame_start, frame_end)]
    angle_data = [None for i in range(frame_start, frame_end)]

    # Apply operations frame-wise
    for i in range(frame_start, frame_end):

      # Determine range-Doppler profile
      (range_data[i], 
       doppler_data[i], 
       doppler_data_sum[i]) = self.range_doppler(radar_data_cube_frames[i],
                                                 'range-velocity-angle', orientation, idx_frame=i, idx_channel_rd_plot=0)

      # TEMP: Plot Range-Doppler map
      # self.plot_range_doppler(doppler_data, 'range-velocity-angle', orientation, idx_frame_plot, idx_channel_rd_plot)

      # Detect targets using detection based on CFAR-CA
      idx_rdm_target_detected, doppler_data_sum_cfar = self.cfar_ca(doppler_data_sum[i], orientation)

      # Group peaks
      detected, rdm_group = self.group_peaks(doppler_data_sum_cfar, size_percent_max=7, orientation=orientation)

      # Apply doppler compensation
      range_aoa = self.doppler_compensation(range_data[i], detected)

      # Apply DOA estimation and determine range-angle maps and point clouds 
      angle_data[i] = self.doa(detected, range_aoa)

      # Display status
      self.print_loop_state(i, frame_end-frame_start, c=0.01)

    # Plot radar signature based on Range-Doppler data
    self.plot_target_signature_rd(doppler_data, angle_data)

    # Plot Range-Doppler map
    self.plot_range_doppler(doppler_data[i], 'range-velocity-angle', orientation, idx_frame_plot, idx_channel_rd_plot)

    # Plot CFAR-processed Range-Doppler map
    self.plot_cfar(doppler_data_sum[i], orientation)

    # Plot group peak-processed Range-Doppler map
    self.plot_group_peaks(rdm_group, orientation, detected)

    # Plot DOA map
    self.plot_doa(angle_data, orientation, detected)

    pass

  def print_loop_state(self, i, n, c=0.1):
    '''Print loop state.
    
    Args:
      i (int):
        Current loop index
      n (int):
        Maximum loops
      c (float):
        Fraction of maximum loops where state is plotted (default: 0.1)

    Returns:

    Raises:
    
    '''
    if n > 10:
      if np.mod(i, np.floor(n*c)) == 0:
        print('Frame loop: ' + str(np.floor(i/n*100.0)) + ' %')
    else:
      print('Frame loop: ' + str(np.floor(i/n*100.0)) + ' %')

  def plot_range_doppler(self, doppler_data, mode, orientation, idx_frame_plot=0, idx_channel_rd_plot=0):
    '''Plot range doppler map.
    
    Args:
      doppler_data (numpy array):
        RD-processed data
      mode (str):
        Either 'range-velocity-angle' (plots afterwards the range-Doppler map) or 'microdoppler' (RD map plotting is surpressed)
      orientation (str):
        Orientation ('azimuth': Azimuth; 'elevation': Elevation)
      idx_frame_plot (int):
        Index of frame (default: 0)
      idx_channel_plot (int):
        Index of channel for RD plot (default: 0)

    Returns:

    Raises:
    '''

    if self.show_range_doppler_map and mode != 'microdoppler':
      if orientation == 'azimuth':
        self.plot_map(self.range_grid, 
                      self.velocity_grid_azimuth, 
                      doppler_data[:, :, idx_channel_rd_plot], 
                      x_label='Velocity / m/s',
                      y_label='Range / m',
                      z_label='Power / 1',
                      title=f'RD / Azimuth / Channel {idx_channel_rd_plot} / Frame {idx_frame_plot}', 
                      mode='2D',
                      extent=None)

      elif orientation == 'elevation':
        self.plot_map(self.range_grid, 
                      self.velocity_grid_elevation, 
                      doppler_data[:, :, 0], 
                      x_label='Velocity / m/s',
                      y_label='Range / m',
                      z_label='Power / 1',
                      title=f'RD / Elevation / Channel {idx_channel_rd_plot} / Frame {idx_frame_plot}', 
                      mode='2D',
                      extent=None)

  def plot_cfar(self, rdm_cfar_ca, orientation):
    '''Plot CFAR-processed range doppler map.
    
    Args:
      rdm_cfar_ca (numpy array):
        RD-processed data
      orientation (str):
        Orientation ('azimuth': Azimuth; 'elevation': Elevation)

    Returns:

    Raises:
    '''

    # Plot base range-Doppler map if desired
    if self.show_cfar_filtered_range_doppler_map:
      if orientation == 'azimuth':

        self.plot_map(self.range_grid, 
                      self.velocity_grid_azimuth, 
                      rdm_cfar_ca, 
                      x_label='Velocity / m/s',
                      y_label='Range / m',
                      z_label='Power / 1',
                      title='RD after CFAR-CA / azimuth', 
                      mode='3D',
                      extent=None)

      elif orientation == 'elevation':

        self.plot_map(self.range_grid, 
                      self.velocity_grid_elevation, 
                      rdm_cfar_ca, 
                      x_label='Velocity / m/s',
                      y_label='Range / m',
                      z_label='Power / 1',
                      title='RD after CFAR-CA / elevation', 
                      mode='3D',
                      extent=None)

  def plot_group_peaks(self, rdm_group, orientation, detected):
    '''Plot group peak-processed range doppler map.
    
    Args:
      rdm_group (numpy array):
        Group peak-processed data
      orientation (str):
        Orientation ('azimuth': Azimuth; 'elevation': Elevation)
      detected (list):
        Peaks (value, index set)

    Returns:

    Raises:
    '''
     
    # Plot base range-Doppler map if desired
    if self.show_cfar_filtered_range_doppler_map:
      if orientation == 'azimuth':
        self.plot_map(self.range_grid, 
                      self.velocity_grid_azimuth, 
                      rdm_group, 
                      x_label='Velocity / m/s',
                      y_label='Range / m',
                      z_label='Power / 1',
                      title='RD / azimuth / grouped', 
                      mode='3D',
                      extent=None,
                      add_points=detected)

      elif orientation == 'elevation':
        self.plot_map(self.range_grid, 
                      self.velocity_grid_elevation, 
                      rdm_group, 
                      x_label='Velocity / m/s',
                      y_label='Range / m',
                      z_label='Power / 1',
                      title='RD / elevation / grouped', 
                      mode='3D',
                      extent=None,
                      add_points=detected)

  def plot_doa(self, angle_fft, orientation, detected):
    ''''Plot DOA map (range, angle).
    
    Args:
      angle_fft (numpy array):
        DOA-processed radar cube
      orientation (str):
        Orientation ('azimuth': Azimuth; 'elevation': Elevation)
      detected (list):
        Peaks (value, index set)
    Returns:

    Raises:
    
    '''
    idx = 1

    # Plot range-map for a specific target
    if self.show_range_angle_map:
      if orientation == 'azimuth':

        self.plot_map(self.range_grid, 
                      self.angle_grid, 
                      angle_fft[:, detected[0]['indices'][0], :], 
                      x_label='Angle / °',
                      y_label='Range / m',
                      z_label='Power / 1',
                      title='RA / azimuth / Target 0', 
                      mode='3D',
                      extent=None)

      elif orientation == 'elevation':

        self.plot_map(self.range_grid, 
                      self.angle_grid, 
                      angle_fft[:, detected[0]['indices'][0], :], 
                      x_label='Angle / °',
                      y_label='Range / m',
                      z_label='Power / 1',
                      title='RA / elevation / Target 0', 
                      mode='3D',
                      extent=None)

      # Plot target's 3D position. TODO: Implement CFAR-CA for better estimation
      target_xyz = self.calculate_target_point(angle_fft, detected)

      self.plot_map(self.range_grid, 
                    self.range_grid, 
                    None, 
                    x_label='x-axis / m',
                    y_label='z-axis / m',
                    z_label='y-xis / m',
                    title='3D position / Target 0', 
                    mode='3D',
                    extent=[],
                    add_points=target_xyz,
                    limits = [[0.0, self.range_grid[-1]],
                              [0.0, self.range_grid[-1]],
                              [0.0, self.range_grid[-1]]])

  def plot_target_signature_rd(self, doppler_data, angle_data):
    '''Plot target signature based on Range-Doppler data.
    
    Args:
      doppler_data (list of Range-Doppler data):
        Range-Doppler data for every frame
      angle_data (list of Range-Doppler-Angle data):
        Range-Doppler-Angle data for every frame

    Returns:

    Raises:
    '''

    # # Calculate differences frame-wise for all channels
    # idx_channel = 0  # TEMP
    # data = [None for i in range(len(doppler_data))]
    # for i in range(len(doppler_data)-1):
    #   data[i] = doppler_data[i+1] - doppler_data[i]
    #   data[i] = np.abs(data[i])
    # data = np.abs(doppler_data)
      
    # # Plot first and last images
    # y_grid = self.range_grid
    # x_grid = self.velocity_grid_azimuth
    # x_label='Velocity / m/s'
    # y_label='Range / m'
    # y_idx_range = [0, 10]
    # fig, ax = plt.subplots(nrows=2 , ncols=3, sharex=False, sharey='all')
    # for i in range(3):
    #   ax[0, i].imshow(data[i][y_idx_range[0]:y_idx_range[1], 
    #                                         :, 
    #                                         idx_channel], 
    #                   extent=[x_grid[0], x_grid[-1], 
    #                          y_grid[-1], y_grid[0]], 
    #                   cmap='viridis', 
    #                   interpolation='nearest', 
    #                   aspect='auto')
    #   #plt.colorbar(ax[0, i], ax=ax, label='Color scale')
    #   ax[0, i].set_xlabel(x_label, fontweight='bold')
    #   ax[0, i].set_ylabel(y_label, fontweight='bold')
    #   ax[0, i].set_title(f"Frame {i}", fontweight='bold')
    # for i in range(3):
    #   idx = len(data) - 1 + i - 3
    #   ax[1, i].imshow(data[idx][y_idx_range[0]:y_idx_range[1], 
    #                                           :, 
    #                                           idx_channel], 
    #                   extent=[x_grid[0], x_grid[-1], 
    #                           y_grid[-1], y_grid[0]], 
    #                   cmap='viridis', 
    #                   interpolation='nearest', 
    #                   aspect='auto')
    #   #plt.colorbar(ax[1, i], ax=ax, label='Color scale')
    #   ax[1, i].set_xlabel(x_label, fontweight='bold')
    #   ax[1, i].set_ylabel(y_label, fontweight='bold')
    #   ax[1, i].set_title(f"Frame {idx}", fontweight='bold')
    


    # Calculate differences frame-wise for all channels
    idx_channel = 0  # TEMP
    # data = [None for i in range(len(angle_data))]
    # for i in range(len(angle_data)-1):
    #   data[i] = angle_data[i+1] - angle_data[i]
    #   data[i] = np.abs(data[i])
    #data = np.abs(angle_data)
    data = np.abs(angle_data)
      
    # Plot first and last images
    y_grid = self.range_grid
    x_grid = self.angle_grid
    x_label='Angle / °'
    y_label='Range / m'
    y_idx_range = [0, 255]
    fig, ax = plt.subplots(nrows=2 , ncols=3, sharex=False, sharey='all')
    for i in range(3):
      ax[0, i].imshow(data[i][y_idx_range[0]:y_idx_range[1], 
                                            idx_channel, 
                                            :], 
                      extent=[x_grid[0], x_grid[-1], 
                             y_grid[-1], y_grid[0]], 
                      cmap='viridis', 
                      interpolation='nearest', 
                      aspect='auto')
      #plt.colorbar(ax[0, i], ax=ax, label='Color scale')
      ax[0, i].set_xlabel(x_label, fontweight='bold')
      ax[0, i].set_ylabel(y_label, fontweight='bold')
      ax[0, i].set_title(f"Frame {i}", fontweight='bold')
    for i in range(3):
      idx = len(data) - 1 + i - 3
      ax[1, i].imshow(data[idx][y_idx_range[0]:y_idx_range[1], 
                                              idx_channel, 
                                              :], 
                      extent=[x_grid[0], x_grid[-1], 
                              y_grid[-1], y_grid[0]], 
                      cmap='viridis', 
                      interpolation='nearest', 
                      aspect='auto')
      #plt.colorbar(ax[1, i], ax=ax, label='Color scale')
      ax[1, i].set_xlabel(x_label, fontweight='bold')
      ax[1, i].set_ylabel(y_label, fontweight='bold')
      ax[1, i].set_title(f"Frame {idx}", fontweight='bold')

    plt.show(block=False)
    pass

  def microdoppler_computation(self, path=None, time_window_length=200, overlap_factor=0.95, pad_factor=4, radar_cube=None, idx_virtual_channel=0, parms=None, bin_indl=10, bin_indu=30, plot=True, orientation='azimuth', antenna_index=0, frame_index=0):
    """Apply signal processing techniques for the determination of the microdoppler map. Note that this works only as intended if one virtual antenna is processed and one chirp is configured for each block so that the time while aggregating blocks is continuous (otherwise, there would be time gaps, each one caused by the other chirps).
    
    Args:
      path (str): 
        File path (default: None)
      time_window_length (int):
        Length of time window (default: 200)
      overlap_factor (float):
        Overlap factor (default: 0.95)
      pad_factor (int):
        Padding factor (default: 4)
      radar_cube (numpy array):
        Radar data cube of one specific frame
      idx_virtual_channel (int):
        Index of virtual channel
      parms (dict):
        Parameters given by the dataset
      bin_indl (int):
        Lower bin index (default: 10)
      bin_indu (int): 
        Upper bin index (default: 30)
      plot (bool):
        PLot results
      orientation (str):
        Orientation: 'azimuth': Azimuth; 'elevation': Elevation (default: None)
      antenna_index (int):
        Index of virtual antenna
      frame_index (int):
        Index of current frame

    Returns:
      img_magnitude_norm (2D numpy array):
        Normalized magnitude of Doppler-time map of microdoppler signature
      img_phase_norm (2D numpy array):
        Normalized phase of Doppler-time map of microdoppler signature

    Raises:
    """

    # Read data saved in the format of the paper of University of Glasgow (2019). If that path is empty, the radar cube of the inital FMCW object is used instead.
    if path is not None:

      # Open file and read data
      file_ID = open(path)
      data_array = file_ID.readlines()
      file_ID.close()

      # Separate parameters and data
      fc = float(data_array[0])
      t_sweep = float(data_array[1])/1000
      nts = int(data_array[2])
      bw = float(data_array[3])
      data_list = data_array[4:]
      
      # Apply preprocessing
      data = [complex(x.replace("i","j")) for x in data_list]
    
    else:

      # Read measurement data from specific board if no frames available
      if self.board_name == 'IWR6843_AOP_REV_G':
        raise NotImplementedError('Data import function for this board not implemented yet.')      # TODO
      elif self.board_name == 'IWR1443_BOOST':
        raise NotImplementedError('Data import function for this board not implemented yet.')      # TODO
      elif self.board_name == 'MMWCAS_RF_DSP':  
        radar_data_cube_frames = self.read_data_ti_mmwcas_dsp(orientation=orientation, 
        n_devices=4, n_tx_all=self.n_tx_all, n_rx_per_device=4,
        adc_calibration_on=True, phase_calibration_only=True,
        dict_file_name_adc_idx_start = {'master': 0, 'slave1': 4, 'slave2': 8, 'slave3': 12}, output_format='3d-sorted')

      # Apply preprocessing
      fc = self.frequency_ref
      t_sweep = self.t_sampling
      nts = self.n_samples
      bw = self.bandwidth
      data = np.reshape(radar_data_cube_frames[frame_index][:, :, antenna_index], newshape=(-1), order='F').tolist()

    num_data_points = len(data)
    prf = 1/t_sweep
    fs = nts/t_sweep
    record_length = num_data_points/nts*t_sweep
    nc = int(record_length/t_sweep)
    overlap_length = round(time_window_length*overlap_factor)
    fft_points = pad_factor*time_window_length
    doppler_bin = prf/fft_points
    doppler_axis = np.arange(-prf/2, prf/2, doppler_bin)
    slope = bw/t_sweep
    whole_duration = nc/prf
    radar_cube = np.reshape(data, (nts, nc), order='F')

    # Apply FFT processing. Part taken from Ancortek code for FFT and IIR filtering. Calculate sweep-specific range profile of 
    # time-series of complex data using shifted FFT.
    if len(radar_cube.shape) == 3:
      nts = np.size(radar_cube[:, :, idx_virtual_channel], 0)
    else:
      nts = np.size(radar_cube, 0)
    win = np.ones((nts, np.size(radar_cube, 1)))
    data_time_weighted = radar_cube*win
    fft_data_time_weigthed = np.fft.fft(data_time_weighted, axis=0)
    tmp = np.fft.fftshift(fft_data_time_weigthed, axes=0)
    data_range_1 = tmp[int(nts/2):nts+1, :]
    ns = int(self.oddnumber(np.size(data_range_1, axis=1)) - 1)
                        
    # Use butterworth filter for filtering range-time-map
    a_filt = np.polymul([1, np.sqrt(2 - np.sqrt(2)), 1], [1, np.sqrt(2 + np.sqrt(2)), 1])
    b_filt = np.zeros_like(a_filt)
    b_filt[-1] = 1

    data_range_mti = np.zeros((np.size(data_range_1, axis=0), ns), dtype=np.complex128)
    for k in range(np.size(data_range_1, axis=0)):
        data_range_mti[k, 0:ns] = sp.signal.lfilter(b_filt, a_filt, data_range_1[k, 0:ns])
    # self.range_freq = np.arange(0,ns)*self.fs/(2*ns)
    # self.range_axis = self.range_freq*self.C_0*self.t_sweep/(2*self.bw)
    data_range_mti = data_range_mti[1:np.size(data_range_mti, axis=0), :]

    #----------------------------------------------
    # Spectrogram processing (Doppler-time-diagram)
    #----------------------------------------------
    self.whole_duration = np.size(data_range_mti, axis=1)/prf
    self.num_segments = int(np.floor((np.size(data_range_mti, axis=1) -\
                                  time_window_length)/np.floor\
                                  (time_window_length*\
                                  (1 - overlap_factor))))
    
    # Perform STFT by applying second (weighted by hanning window) FFT on sweep-related range
    # profile to get spectrogram (time-related distribution of frequency components) which is 
    # the doppler-time map.
    self.data_spec_mti2 = 0
    self.data_spec_mti3 = 0
    self.data_spec2 = 0
    for rbin in range(bin_indl-1,bin_indu):
        f,t,time_freq = self.myspecgram(data_range_mti[rbin,:],
                                        fft_points,
                                        fs,
                                        time_window_length,
                                        overlap_length)
        data_mti_temp = np.fft.fftshift(time_freq,axes=0)
        self.data_spec_mti2 = self.data_spec_mti2 + \
                              np.abs(data_mti_temp)
        self.data_spec_mti3 = self.data_spec_mti3 + \
                              np.arctan2(np.imag(data_mti_temp),
                                          np.real(data_mti_temp))
    
    time_axis = np.linspace(0, self.whole_duration,np.size(self.data_spec_mti2, axis=1))
    img_mag = 20*np.log10(np.abs(self.data_spec_mti2))
    img_pha = np.abs(self.data_spec_mti3)
    
    # Preprocess (Normalization, threshold filtering, SVD for data 
    # compression)
    img_magnitude_norm = self.normalize(img_mag)
    img_phase_norm = self.normalize(img_pha)

    # Plot results if desired
    if plot:
        _, ax = plt.subplots(2,1)
        extent = [time_axis[0],
                  time_axis[-1],
                  doppler_axis[0]*self.C/2/fc,
                  doppler_axis[-1]*self.C/2/fc]
        
        ax[0].imshow(X=img_magnitude_norm, 
                    extent=extent, 
                    aspect="auto", 
                    cmap='jet')
        ax[0].set_ylim((-6,6))
        ax[0].set_xlabel('Time (t) / s')
        ax[0].set_ylabel('Doppler (v) / m/s')
        ax[0].set_title('Log. magnitude (normalized)')

        ax[1].imshow(X=img_phase_norm, 
                    extent=extent, 
                    aspect="auto", 
                    cmap='jet')
        ax[1].set_xlabel('Time (t) / s')
        ax[1].set_ylabel('Doppler (v) / m/s')
        ax[1].set_title('Phase (normalized)')

    # # Show time-doppler map for saving
    # extent = [time_axis[0],
    #           time_axis[-1],
    #           doppler_axis[0]*self.C/2/fc,
    #           doppler_axis[-1]*self.C/2/fc]
    
    # fig, ax1 = plt.subplots()
    # dt_map = plt.imshow(img_mag, extent=extent, aspect="auto", cmap='jet')
    # clim = dt_map.get_clim()
    # dt_map.set_clim(clim[1] - 70, clim[1])
    # plt.ylim(-6,6)
    # plt.axis('off')

    return img_magnitude_norm, img_phase_norm

  def simulation_data_to_radar_cube_mimo(self, orientation):
    """Simulate MIMO case.
    
    Simulate MIMO case using specific targets, antenna and signal processing configuration for one frame.
    
    Args:
      orientation (str):
        Orientation: 'azimuth': Azimuth; 'elevation': Elevation
    
    Returns:
      radar_data_cube_frames (list):
        Frames containing simulated data (radar cube)

    Raises:
    """
      
    if orientation == 'azimuth':
      n_virtual_antennas_selected = self.n_virtual_antennas_selected_azimuth
      n_blocks = self.n_blocks_azimuth
      chirp_schedule = self.chirp_schedule_azimuth
      t_block = self.t_block_azimuth
    elif orientation == 'elevation':
      n_virtual_antennas_selected = self.n_virtual_antennas_selected_elevation
      n_blocks = self.n_blocks_elevation
      chirp_schedule = self.chirp_schedule_elevation
      t_block = self.t_block_elevation

    # Initialize data matrices
    radar_data_cube_frames = [np.zeros((self.n_samples, 
                                        n_blocks, 
                                        n_virtual_antennas_selected), dtype=np.complex128) for i in range(self.n_frames)]
    
    # Get relative time vector of sampling period
    t_sampling_rel = np.arange(0, self.n_samples)*self.t_sample

    # Calculate positions of TX and RX antennas based on selected orientation
    coordinates_tx = np.zeros((n_virtual_antennas_selected, 3))
    coordinates_rx = np.zeros((n_virtual_antennas_selected, 3))
    for i in range(n_virtual_antennas_selected):
      idx_tx_rx = self.vx_azimuth_selected[i][1]
      coordinates_tx[i, :] = np.append(self.tx_pos_abs[idx_tx_rx[0]], 0.0)
      coordinates_rx[i, :] = np.append(self.rx_pos_abs[idx_tx_rx[1]], 0.0)

    # Get number of targets
    n_targets = len(self.targets_sim_init)

    # Get number of chirps per block
    n_chirps_per_block = len(chirp_schedule)

    # Apply for all frames, blocks, chirps and virtual channels
    for g in range(self.n_frames):
      for h in range(n_blocks):
        for i in range(n_chirps_per_block):
          for k in chirp_schedule[i]:

            # Get absolute time vector and simulate targets to get their position based on the given time vector
            t_sampling_abs = g*self.t_frame + h*t_block + i*self.t_chirp + t_sampling_rel
            target_sim_pos = self.simulate_target_position_velocity(t_sampling_abs)

            # Get beat signals for every target and superpose
            x_IF_i = np.zeros((self.n_samples, n_targets), dtype=np.complex128)
            for j in range(n_targets):

              # Get distances from TX and RX antenna to target respectively and determine beat signal
              distances_tx_target, distances_rx_target = self.get_distances_rx_tx_to_target(target_sim_pos, coordinates_tx[k], coordinates_rx[k],  j)
              x_IF_i[:, j] = self.simulate_sampled_beat_signal(t_sampling_rel, distances_tx_target, distances_rx_target, j)

            # Sum up all target-related IF signals
            x_IF = np.sum(x_IF_i, axis=1)

            # Save to radar cube
            radar_data_cube_frames[g][:, h, k] = x_IF

          # Show status
          if (g*n_blocks*n_chirps_per_block + h*n_chirps_per_block + i) % round((self.n_frames*n_blocks*n_chirps_per_block)*0.1) == 0:
            print(f"Simulation status: {(g*n_blocks*n_chirps_per_block + h*n_chirps_per_block + i) / (self.n_frames*n_blocks*n_chirps_per_block)*100:.0f} %")
    
    return radar_data_cube_frames

  def sar_processing_chain(self, folder_path, json_file_path, calibration_file_path):
      """Apply SAR processing chain.
      
      Data is read and provided in 3D (n_samples x n_blocks x (n_chirps(n_TX_all if a simple TDM-MIMO approach) x n_rx_all)) or 4D (n_samples x n_blocks x n_rx_all x n_tx), where the order of TX and RX is ascending. In order to apply classical MIMO methods, the ordering has to be done based on virtual channel configuration to yield a contiguous order (left to right). In this version, the data has to be collected with a 
      azimuth setup of the radar HW.

      Args:
      folder_path (str):
        Path to folder containing data
      json_file_path (str):
        Path to JSON file containing parameters
      calibration_file_path (str):
        Path to calibration files
      
      Returns:
        sar_image (numpy array):
          SAR image

      Raises:
      """

      # Read data
      # HACK!
      dict_file_name_adc_idx_start = {'master': 0, 
                                      'slave1': 4, 
                                      'slave2': 8, 
                                      'slave3': 12}
      # dict_file_name_adc_idx_start = {'master': 12, 
      #                                 'slave1': 8, 
      #                                 'slave2': 4, 
      #                                 'slave3': 0}
      adc_out = self.read_data_ti_mmwcas_dsp(orientation='azimuth', 
                                             n_devices=4, n_tx_all=self.n_tx_all, n_rx_per_device=4, 
                                             adc_calibration_on=False, phase_calibration_only=True,
                                             dict_file_name_adc_idx_start= dict_file_name_adc_idx_start,
                                             output_format='3d-sorted')
      
      # Apply SAR method
      sar_image = self.sar_rma(adc_out)

      return sar_image

  # endregion

# region Test functions

def test_radar_data_simulation(device_index=None, parms=None):
  ''' Test radar simulation data.
  
  Args:
    device_index (str):
      Device index(default: None)
    parms (dict): 
      Parameters (default: None)

  Returns:

  Raises:
  '''

  fmcw_1 = Fmcw(parms['parms_board'][device_index], 
                          parms['parms_processing'], 
                          parms['parms_chirp_profile'],
                          parms['parms_target'],
                          parms['parms_show'],
                          measurement_folder_path=None,
                          parameter_file_path=None,
                          calibration_file_path=None)
    
  radar_cube = fmcw_1.simulation_data_to_radar_cube_mimo('azimuth')
  fmcw_1.range_velocity_angle_computation(radar_cube, 'azimuth', idx_frame_plot=0, idx_channel_rd_plot=0)


def test_calibration_file_generation(device_index=None, parms=None):
  '''Test calibration file geenration using MATLAB from TI.
  
  Args:
    device_index (str):
      Device index (default: None)
    parms (dict): 
      Parameters (default: None)

  Returns:

  Raises:
  '''

  fmcw_1 = Fmcw(parms['parms_board'][device_index], 
                parms['parms_processing'], 
                parms['parms_chirp_profile'],
                parms['parms_target'],
                parms['parms_show'],
                measurement_folder_path=None,
                parameter_file_path=None,
                calibration_file_path=None)
  
  eng = matlab.engine.start_matlab()
  eng.addpath('mmWave_MMWCAS_RF-DSP_EVM\\Calibration\\Processing_Calibration_MATLAB')
  eng.cascade_MIMO_antennaCalib(nargout=0)


def test_measurement_parameters(device_index=None, parms=None):
  '''Test calibration file geenration using MATLAB from TI.
  
  Args:
    device_index (str):
      Device index (default: None)
    parms (dict): 
      Parameters (default: None)

  Returns:

  Raises:
  '''

  fmcw_1 = Fmcw(parms['parms_board'][device_index], 
                parms['parms_processing'], 
                parms['parms_chirp_profile'],
                parms['parms_target'],
                parms['parms_show'],
                measurement_folder_path=None,
                parameter_file_path=None,
                calibration_file_path=None)
  
  n_total_frames = fmcw_1.estimate_number_of_frames_for_stroke(n_chirps=12, n_blocks=64, t_idle=5e-6, t_ramp=40e-6, t_interframe=100e-3, t_stroke=5.08)      
  n_megabytes_estimated = fmcw_1.estimate_megabyte_total(n_devices=4, n_rx_per_device=4, n_samples=256, n_chirps=12, n_blocks=64, n_frames=n_total_frames, n_byte_per_sample=2)


def test_standard_tdm_mimo_processing(device_index=None, parms=None):
  '''Test standard TDM-MIMO processing chain (range, velocity, angle).
  
  Args:
    device_index (str):
      Device index (default: None)
    parms (dict): 
      Parameters (default: None)

  Returns:

  Raises:
  '''

  # fmcw_1 = Fmcw(parms['parms_board'][device_index], 
  #               parms['parms_processing'], 
  #               parms['parms_chirp_profile'],
  #               parms['parms_target'],
  #               parms['parms_show'],
  #               parms['parms_sar'],
  #               measurement_folder_path='measurements\\bore_static_target_2025-05-08',
  #               parameter_file_path='measurements\\bore_static_target_2025-05-08\\Cascade_Capture.mmwave.json',
  #               calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\Calibration\calibrateResults_high.mat')
  
  # fmcw_1 = Fmcw(parms['parms_board'][device_index], 
  #               parms['parms_processing'], 
  #               parms['parms_chirp_profile'],
  #               parms['parms_target'],
  #               parms['parms_show'],
  #               parms['parms_sar'],
  #               measurement_folder_path='measurements\\bore_moving_target_2025-05-08',
  #               parameter_file_path='measurements\\bore_moving_target_2025-05-08\\Cascade_Capture.mmwave.json',
  #               calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\Calibration\calibrateResults_high.mat')

  fmcw_1 = Fmcw(parms['parms_board'][device_index], 
                parms['parms_processing'], 
                parms['parms_chirp_profile'],
                parms['parms_target'],
                parms['parms_show'],
                parms['parms_sar'],
                measurement_folder_path='measurements\\bore_moving_target_2_2025-05-08',
                parameter_file_path='measurements\\bore_moving_target_2_2025-05-08\\Cascade_Capture.mmwave.json',
                calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\Calibration\calibrateResults_high.mat')

  # fmcw_1 = Fmcw(parms['parms_board'][device_index], 
  #               parms['parms_processing'], 
  #               parms['parms_chirp_profile'],
  #               parms['parms_target'],
  #               parms['parms_show'],
  #               parms['parms_sar'],
  #               measurement_folder_path='measurements\\angled_static_target_2025-05-08',
  #               parameter_file_path='measurements\\angled_static_target_2025-05-08\\Cascade_Capture.mmwave.json',
  #               calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\Calibration\calibrateResults_high.mat')

  # fmcw_1 = Fmcw(parms['parms_board'][device_index], 
  #               parms['parms_processing'], 
  #               parms['parms_chirp_profile'],
  #               parms['parms_target'],
  #               parms['parms_show'],
  #               parms['parms_sar'],
  #               measurement_folder_path='measurements\\angled_static_target_2_2025-05-08',
  #               parameter_file_path='measurements\\angled_static_target_2_2025-05-08\\Cascade_Capture.mmwave.json',
  #               calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\Calibration\calibrateResults_high.mat')

  fmcw_1.range_velocity_angle_computation(None, 'azimuth', idx_frame_plot=0, idx_channel_rd_plot=0)
  pass


def test_microdoppler_processing(device_index=None, parms=None):
  '''Test microdoppler processing chain.
  
  Args:
    device_index (str):
      Device index (default: None)
    parms (dict): 
      Parameters (default: None)

  Returns:

  Raises:
  '''

  fmcw_1 = Fmcw(parms['parms_board'][2], 
                parms['parms_processing'], 
                parms['parms_chirp_profile'],
                parms['parms_target'],
                parms['parms_show'],
                parms['parms_sar'],
                measurement_folder_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\\Data\\Calibration_1',
                parameter_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\\Data\\Calibration_1\\Cascade_Capture.mmwave.json',
                calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\Calibration\calibrateResults_high.mat')

  (img_magnitude_norm, 
  img_phase_norm) = fmcw_1.microdoppler_computation(path='frameworks and toolboxes\\dataset_glasgow 2019_radhar\\01.12.2017\\1P36A01R01.dat', antenna_index=0, frame_index=0)
  # (img_magnitude_norm, 
  # img_phase_norm) = fmcw_1.microdoppler_computation(path=None, antenna_index=0, frame_index=0)


def test_sar_processing(device_index=None, parms=None, measurement_path=None):
  '''Test SAR processing chain.
  
  Args:
    device_index (str):
      Device index (default: None)
    parms (dict): 
      Parameters (default: None)

  Returns:

  Raises:
  '''

  if measurement_path is None:
    # measurement_path = 'Cascade_Capture_WnukSAR\\Messung_mit_Bewegung_ohne_Objekt_2019-03-26'
    #measurement_path = 'Cascade_Capture_WnukSAR\\Cascade_Capture_Test_8'
    # measurement_path = 'Cascade_Capture_WnukSAR\\measurement_results_25_03_2025\\Messung_mit_Objekt_mit_Bewegung2\\Cascade_Capture'
    # measurement_path = 'Cascade_Capture_WnukSAR\\measurement_results_25_03_2025\\Messung_mit_Objekt_mit_Bewegung3\\Cascade_Capture'
    # measurement_path = 'Cascade_Capture_WnukSAR\\measurement_results_25_03_2025\\Messung_ohne_Objekt_mit_Bewegung\\Cascade_Capture'
    measurement_path = 'Cascade_Capture_WnukSAR\\measurement_results_25_03_2025\\Messung_ohne_Objekt_mit_Bewegung3\\Cascade_Capture'

  fmcw_1 = Fmcw(parms['parms_board'][device_index], 
                parms['parms_processing'], 
                parms['parms_chirp_profile'],
                parms['parms_target'],
                parms['parms_show'],
                parms['parms_sar'],
                measurement_folder_path=measurement_path,
                parameter_file_path=measurement_path + '\\Cascade_Capture.mmwave.json',
                calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\\Calibration\\calibrateResults_high_20240829T112982_Calibration_1.mat')

  image = fmcw_1.sar_processing_chain(folder_path=measurement_path,
                                      json_file_path=measurement_path + '\\Cascade_Capture.mmwave.json',
                                      calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\\Calibration\\calibrateResults_high_20240829T112982_Calibration_1.mat')


def test_measure_and_standard_tdm_mimo_processing(device_index=None, parms=None):
  '''Test measurement and standard TDM-MIMO processing chain (range, velocity, angle).
  
  Args:
    device_index (str):
      Device index (default: None)
    parms (dict): 
      Parameters (default: None)

  Returns:

  Raises:
  '''

  # Send LUA script for initial configuration using a MATLAB script (workaround)
  print('Configuring FMCW sensor...')
  me = matlab.engine.start_matlab()
  me.cd(os.getcwd(), nargout=0)
  _ = me.config_mmwstudio()
  print('Done.')

  inf_loop = True
  while inf_loop:

    # Execute measurement
    print('FMCW measurement starting....')
    results = me.measure_mmwstudio()
    print('Done.')

    # Apply processing
    fmcw_1 = Fmcw(parms['parms_board'][device_index], 
                  parms['parms_processing'], 
                  parms['parms_chirp_profile'],
                  parms['parms_target'],
                  parms['parms_show'],
                  parms['parms_sar'],
                  measurement_folder_path='measurements\\bore_static_target_2025-05-08',
                  parameter_file_path='measurements\\bore_static_target_2025-05-08\\Cascade_Capture.mmwave.json',
                  calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\\Calibration\\calibrateResults_high.mat')
    fmcw_1.range_velocity_angle_computation(None, 'azimuth', idx_frame_plot=0, idx_channel_rd_plot=0)

    # Ask user if new cycle desired
    ans = input("New cycle? y/n")
    if ans=='y':
      inf_loop = True
    else:
      inf_loop = False
      print('Application closed.')

def export_signature_image_standard_tdm_mimo_processing(device_index=None, parms=None, path_folder_data=None):
  '''Export signature image using standard TDM-MIMO processing chain (range, velocity, angle).
  
  Args:
    device_index (str):
      Device index (default: None)
    parms (dict): 
      Parameters (default: None)
    path_folder_data (str):
      Path to FMCW radar data

  Returns:

  Raises:
  '''

  fmcw_1 = Fmcw(parms['parms_board'][device_index], 
                parms['parms_processing'], 
                parms['parms_chirp_profile'],
                parms['parms_target'],
                parms['parms_show'],
                parms['parms_sar'],
                measurement_folder_path=path_folder_data,
                parameter_file_path=path_folder_data + '\\Cascade_Capture.mmwave.json',
                calibration_file_path='frameworks and toolboxes\\mmWave_MMWCAS_RF-DSP_EVM\Calibration\calibrateResults_high.mat')

  fmcw_1.range_velocity_angle_computation(None, 'azimuth', frame_range=[0, 10],idx_frame_plot=0, idx_channel_rd_plot=0)
  pass

   
# endregion


if __name__ == '__main__':

  print(f'====')
  print(f'Fmcw')
  print(f'====')

  # region Parameter

  # Select device index:
  #   0: 'IWR6843_AOP_REV_G'
  #   1: 'IWR443_BOOST'
  #   2: 'MMWCAS_RF_EVM'
  device_index = 2

  # # Save to JSON file
  # with open('parms_fmcw.json', 'w') as json_file:
  #     json.dump(parms_fmcw, json_file, indent=4)

  # Load variables from the JSON file
  with open('parms_fmcw.json', 'r') as file:
      parms_fmcw = json.load(file)

  # endregion

  # region Application

  # (1) Test radar data simulation
  # test_radar_data_simulation(device_index, parms_fmcw)

  # (2) Run calibration file generation function using MATLAB files from TI
  # test_calibration_file_generation(device_index, parms_fmcw)

  # (3) Test measurement parameters
  # test_measurement_parameters(device_index, parms_fmcw)

  # (4) Test standard TDM-MIMO processing chain (range, velocity, angle)
  # test_standard_tdm_mimo_processing(device_index, parms_fmcw)
  export_signature_image_standard_tdm_mimo_processing(device_index, parms_fmcw, 'C:\\WORK\\CAISA\Promotion\\4 - Entwicklung\Software\Python\\Inline Food Inspection\\measurements_ifi\\2025_04_25\\anomalous\\flat_mtl_3mm_0deg\\fmcw')

  # (5) Test microdoppler computation
  # test_microdoppler_processing(device_index, parms_fmcw)

  # (6) Test SAR processing chain
  # test_sar_processing(device_index, parms_fmcw, measurement_path=None)

  # (7) Test radar data interpretation
  # test_measure_and_standard_tdm_mimo_processing(device_index, parms_fmcw)

  # endregion

  pass