# IFI
## Toolbox and application for THz and optical camera sensing for Inline Food Inspection
This project containes all the relevant files for the implementation of the Inline-Food-Inspection application based on the sub-terahertz sensing system 'TeraSense TeraFAST-256-HS-100' and the industrial-grade optical camera 'Basler daA3840-45uc'.

The following files and provided:
- IFIMeasure.py: Class file that defines the IFIMeasurement class and a main call (at the end of the file) for the asynchronous call of the sensing systems. Requires the classes 'THz' in thz.py and 'Camera' in Camera.py to work as intended. Remove FMCW-related operations.
- IFIAnalysis.py: Class file that defines the IFIAnalysis class, which is required to apply the postprocessing and evaluation operations.
- edge_detection_udp.py: File that has to be implemented on a Raspberry Pie (> 4.0) in order to send a trigger via UDP message to the recipient (laptop running IFI).
- IFIDiagrams.ipynb: Jupyter Notebook file that plots the diagrams that were used in the paper 'Detection of Low-Density Foreign Objects in Confectionery Products Using Sub-Terahertz Technology'.
- conveyor_belt_drive_control_siemens_s7_1200.zip: TIA-program of the conveyor belt drive based on the PLC SIEMENS S7-1200.
- requirements.txt: Requirements file for the installation of required external frameworks using PIP.
- thz_processed_image_extracts.zip: Dataset that was used for the paper 'Detection of Low-Density Foreign Objects in Confectionery Products Using Sub-
Terahertz Technology' consisting of THz images for the selected confectionery products (four types) using different foreign objects (that vary in material and sizes) and a varying attenuation. 

Branch: Master
