from pypylon import pylon
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


class Camera:
    def __init__(self):
        pass

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
      camera.TriggerMode = 'On'
      camera.TriggerSource = 'Software'

      return camera
    
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
      print(f"Gain: {camera.Gain.Value}")
      print(f"Trigger selector: {camera.TriggerSelector.Value}")
      print(f"(Symbolics of trigger selector: {camera.TriggerSelector.Symbolics})")
      print(f"Pixel format: {camera.PixelFormat.Value}")
      print(f"Exposure time: {camera.ExposureTime.Value}")

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
    
    def basler_camera_disconnect(self, camera):
        '''Disconnect from Basler camera.'''
        camera.Close()
        return camera
