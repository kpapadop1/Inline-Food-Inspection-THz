# IFI
## Toolbox and application for THz, FMCW and MV measurements for Inline Food Inspection
This project containes all the relevant files for the implementation of the Inline-Food-Inspection application. Note, that the two class files for the call of THz and Machine Vision measurements have to be downloaded from the corresponding github project (files to be placed in the same folder as IFI.py).

The following files and provided:
- IFI.py: containes the IFI class and the main call. Refers to the classes 'THz' in thz.py and 'MV' in MV.py.
- configmmwstudio.m: Matlab-file for the configuration of mmwave Studio during startup sequence of the main IFI call. Note that mmstudio has to be started manually before this function is called.
- edge_detection_udp.py: File that has to be implemented on a Raspberry Pie (> 4.0) in order to send a trigger via UDP message to the recipient (laptop running IFI).
- requirements.txt: Requirements file for the installation of required external frameworks using PIP.
- startmmwstudio.m: Matlab-file for the execution of a measurement (burst of frames). Data is saved locally on the MMWCAS-EVM-Board and transferred to the local TI folder (post processing inside of mmw studio section). Note that a connection via WinSCP has to be established to the backend board and after every measurement the folders have to be cleared in order to avoid redundant data transfer.

Branch: Master
