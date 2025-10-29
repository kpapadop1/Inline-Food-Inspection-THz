# IFI
## Toolbox and application for sub-THz, optical and FMCW-based sensing for Inline Food Inspection
This project containes all the relevant files for the implementation of the Inline-Food-Inspection application based on the sub-terahertz sensing system 'TeraSense TeraFAST-256-HS-100', the industrial-grade optical camera 'Basler daA3840-45uc, and the FMCW-based radar system 'TI MMWCAS-RF/DSP'

The following files and folders and provided:

- (1) IFI.py: Class file that defines the IFI class and a main call (at the end of the file) for the asynchronous call of the measurement functions of all related sensing systems as well as all postprocessing and evaluation operations. Requires the files in the folder 'auxiliary', if all functions are used.
- edge_detection_udp.py: File that has to be implemented on a Raspberry Pie (> 4.0) in order to send a trigger via UDP message to the recipient (laptop running IFI).
- (2) IFIDiagrams.ipynb: Jupyter Notebook file that plots the diagrams that were used in the paper 'Detection of Low-Density Foreign Objects in Confectionery Products Using Sub-Terahertz Technology'.
- conveyor_belt_drive_control_siemens_s7_1200.zip: TIA-program of the conveyor belt drive based on the PLC SIEMENS S7-1200.
- (3) Folder 'auxiliary': Containes classes for THz measurement (based on TeraSense's Toolbox), optical camera (based on Basler's Toolbox), and FMCW-based radar system (requiring MATLAB)
- (4) thz_processed_image_extracts.zip: Dataset that was used for the paper 'Detection of Low-Density Foreign Objects in Confectionery Products Using Sub-
Terahertz Technology' consisting of THz images for the four selected types of confectionery products using foreign objects of various materials and sizes as well as varying attenuation. Extract contents to same level as the IFI script.
- (5) requirements.txt: Requirements file for the installation of all required external frameworks using PIP package installer.