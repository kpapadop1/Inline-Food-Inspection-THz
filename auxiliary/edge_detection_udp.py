# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MAIN-SCRIPT
  This script applies the operations on the Rapberry Pi in order to send a message string to a pc via Ethernet using the protocol UDP.
  The message is supposed to trigger the IFI measurements.
  
SYNTAX
  -

INPUT VARIABLES
  -

OUTPUT VARIABLES
  -

DESCRIPTION
  This script applies the operations on the Rapberry Pi in order to send a message string to a pc via Ethernet using the protocol UDP.
  The message is supposed to trigger the IFI measurements.


SEE ALSO
  -
  
FILE
  .../edge_detection_udp.py

ASSOCIATED FILES
  -

AUTHOR(S)
  R. Wnuk

DATE
  2024-12-19

LAST MODIFIED
  2025-02-16      K. Papadopoulos      Optimized script

V1.0 / Copyright 2022 - Konstantinos Papadopoulos
-------------------------------------------------------------------------------
Notes
------
pip install pigpio

Todo:
------
  
-------------------------------------------------------------------------------
"""
#==============================================================================
#%% DEPENDENCIES
#==============================================================================

# import packages
import socket
import time
from datetime import datetime
import pigpio

#==============================================================================
#%% CONSTANTS
#==============================================================================
# Set GPIO
GPIO_Pin = 17

# IP and port of this device (Raspberry Pie)
UDP_IP = "192.168.33.2"
UDP_Port = 12345

# Waiting time in seconds
T_WAIT = 1.0

#==============================================================================
#%% APPLICATION
#==============================================================================

# Establish UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define funtion for the setup and send operation of the message
def falling_edge(gpio, level, tick):
    message = f"trigger, {time.time()}"
    sock.sendto(message.encode('utf-8'), (UDP_IP, UDP_Port))
    print(f"Message '{message}' sent to {UDP_IP};{UDP_Port}.")

# Handle to access the local Raspberry Pi's GPIO
pi = pigpio.pi()

# Abort operation if device is not connected
if not pi.connected:
    print("Error: Connection unsuccessful.")
    exit()

# Set pin to input
pi.set_mode(GPIO_Pin, pigpio.INPUT)

# Callback for edge detection
cb = pi.callback(GPIO_Pin, pigpio.FALLING_EDGE, falling_edge)

# Main program
try:
    print("Press CTRL+C to abort execution.")
    while True:
        time.sleep(T_WAIT)
except KeyboardInterrupt:
    print("\n Program closed.")
finally:
    # Clear ressources
    cb.cancel()
    pi.stop()