import time
import threading
import sys

import matplotlib.pyplot as plt

import cv2
import numpy as np

from teraFAST.processor import processor as Processor
from teraFAST.worker    import Worker

class THz:
    """
    Class for live streaming and conveyor capture (merged images) with option to make the live stream invisible while the capture continues to work. The number of images to be captured can be configured via the class parameter "num_shots".
    """
    def __init__(self, parms_thz):
        """
        Initialization: Set processor instance, image sizes, worker and capture parameters.

        Args:
            parms_thz (dict):   All THz parameters.

        Returns:

        Raises:
            RuntimeError:       If TeraFast processor can't be initialized.
        """

        # Set core parameters
        self.belt_speed         = parms_thz['init']['BELT_SPEED']
        self.chunk_size         = parms_thz['init']['FRAME_LENGTH']
        self.aspect_ratio       = parms_thz['init']['ASPECT_RATIO']
        self.livestream_visible = parms_thz['init']['LIVESTREAM_VISIBLE']
        self.num_shots          = parms_thz['n_images']



        # Status flags
        self.should_stop  = False
        self.capture_mode = False
        self.raw          = True

        self.X_SIZE = int(self.chunk_size)
        self.Y_SIZE = int(self.chunk_size * self.aspect_ratio)

        MAX_VALUE = (1<<15)-1
        self.mask = 1e99*np.ones((self.X_SIZE,self.Y_SIZE),dtype=float)
        self.norm = np.ones((self.X_SIZE,self.Y_SIZE),dtype=float)*MAX_VALUE
        self.normSrc = np.ones((self.X_SIZE,self.Y_SIZE),dtype=float)*MAX_VALUE
        self.normNone = self.normSrc = np.ones((self.X_SIZE,self.Y_SIZE),dtype=float)*MAX_VALUE
        self.mask = self.maskNone = 1e99*np.ones((self.X_SIZE,self.Y_SIZE),dtype=float)
        self.threshold = 3.0
        self.recordingCount = 30


        # Instantiate processor
        try:
            self.p = Processor(rows=self.chunk_size, threaded=True, raw_mode=True)
        except RuntimeError as e:
            raise RuntimeError(f"Fehler bei der Initialisierung des TeraFAST-Processors: {e}")

        # Set image and recording parameters
        self.rows = self.p.Y_SIZE
        self.cols = self.p.X_SIZE
        self.p.SetDifference()
        self.p.SetRate(parms_thz['init']['RATE'])
        self.img_size = (self.rows, self.cols)

        # Instantiate worker for image generation
        self.w = Worker(self.img_size)

        # Calculate and set capture parameters
        self.capture_delay      = self.rows / self.belt_speed
        self.capture_count      = 0
        self.next_capture_time  = 0.0
        self.captured_frames    = []
    
        self.SetNorm()
        
    
    def SetNorm(self,data=None, mask=None):
        """Sets current normalization data to be used in processing.
        Parameters:
        data    -   numpy.ndaray with data to be used for normalization (default:None).
        mask    -   numpy.ndaray with corresponding mask data to be used for normalization (default:None).
        Returns:
        None
        """
        if data is None:
            self.norm = self.normNone
            self.normSrc = self.normNone
            self.mask = self.maskNone
            self.prepareNorm()
            
            return
        if ( len(data) != self.X_SIZE  ) :
            self.norm = self.normNone
            self.normSrc = self.normNone
            self.mask = self.maskNone
            raise RuntimeError("SetNorm::Range mismatch")
            return
        else:
            self.normSrc[:,:] = data
        if mask is not None:
            try:
                self.mask[:,:] = mask.copy()
            except:
                self.mask[:,:] = np.array(mask,dtype=float)            
        else:
            self.mask = self.maskNone

        self.prepareNorm()


    def prepareNorm(self):
        """Prepare data to be used in normalization.

        Returns:
            None
"""
        try:
            self.norm = self.normSrc.copy()
        except:
            self.norm = np.array(self.normSrc,dtype=float)

        self.norm = np.where(  self.mask >  self.threshold,self.norm, np.inf )
        print("Norm prepared.")

    def start_stream(self):
        """
        Starts the processor stream and optionally the console thread. Must be called before execute_measurement_simple().

        Args:

        Returns:

        Raises:
        """

        if self.livestream_visible:
            threading.Thread(target=self._console_input_loop, daemon=True).start()
        self.p.start(self.on_result, self.on_error)

    def terasense_stop(self,source):
        """
        Stops the processor stream and closes windows.

        Args:

        Returns:

        Raises:
        """

        self.p.stop()
        if self.livestream_visible:
            cv2.destroyAllWindows()
        print("Stream stopped.")

    def start_capture_sequence(self):
        """
        Initializes capture state.

        Args:

        Returns:

        Raises:
        """

        self.capture_mode      = True
        self.capture_count     = 0
        self.next_capture_time = time.time()
        self.captured_frames   = []
        print(f"[CAPTURE] sequence started (Δt = {self.capture_delay:.3f} s, shots = {self.num_shots})")

    def on_result(self, data):
        """
        Callback for new frames: shows LiveView and captures images if desired.

        Args:
            data:       Raw THz data

        Returns:

        Raises:
        """

        if data is None:
            return
        self.raw = data.copy()
        data = self.p.SubstractBG(data)
        data = self.p.Normalize(data)
        img = self.w.makeImg(data)
        img_rot =  cv2.flip(img, 0)
        
        bgr = cv2.cvtColor(img_rot, cv2.COLOR_RGB2BGR)

        if self.livestream_visible:
            cv2.imshow("TeraFAST Live View", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.should_stop = True
            elif key == ord('m') and not self.capture_mode:
                self.start_capture_sequence()

        if self.capture_mode and time.time() >= self.next_capture_time:
            self.captured_frames_raw.insert(0, self.raw.copy())
            self.captured_frames.insert(0, img.copy())
            self.capture_count += 1
            self.next_capture_time += self.capture_delay
            print(f"[CAPTURE] Image {self.capture_count}/{self.num_shots}")
            if self.capture_count >= self.num_shots:
                self.capture_mode = False
                self.should_stop = True

    def on_error(self, err):
        """
        Callback for device errors.

        Args:
            err:       Error message string

        Returns:

        Raises:
        """

        print("Error from device:", err)
        self.should_stop = True
        self.capture_mode = False

    def _console_input_loop(self):
        """
        Reads console commands: 'm', 'q'

        Args:

        Returns:

        Raises:
        """

        print("Console mode: 'm'=Capture, 'q'=Exit")
        while not self.should_stop:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 'm' and not self.capture_mode:
                self.start_capture_sequence()
            elif cmd == 'q':
                self.should_stop = True

    def execute_measurement_simple(self):
        """
        Trigger only captures 'num_shots' frames. The stream must first be started with 'start_stream()'.

        Args:

        Returns:

        Raises:
            RuntimeError:       If stream wasn't started.
        """

        if not hasattr(self, 'p'):
            raise RuntimeError("Stream wasn't started. Call 'start_stream()' first.")

        # Reset flags to wait in capture mode
        self.should_stop      = False
        self.capture_mode     = True
        self.capture_count    = 0
        self.captured_frames  = []
        self.captured_frames_raw = []
        self.next_capture_time = time.time()

        # Wait until completion
        while self.capture_mode and not self.should_stop:
            time.sleep(0.01)

        # Check images and stitch together
        if len(self.captured_frames) != self.num_shots:
            raise RuntimeError(f"Expects {self.num_shots} frames. Received {len(self.captured_frames)} frames.")
        combined = cv2.vconcat(self.captured_frames)
        combined_raw = cv2.vconcat(self.captured_frames_raw)
        print("Capture completed.")

        return combined, combined_raw
    
    #static 
    def bg_subtraction(raw_data):
        #correct overflow error
        for i in range(0, len(raw_data)):
            for j in range(0, len(raw_data[i])):
                if raw_data[i][j] >= 30000:
                    raw_data[i][j] = raw_data[i][j] - 65536
        
        #calculate the the mean background from the last lines 
        last_lines = raw_data[:,0]
        #mean_last_lines = np.mean(last_lines, axis=1,keepdims=True)
        sub_array = last_lines -(-32767.0)
        subfield = np.empty((256,2048))
        for i in range(0,2048):
            subfield[:,i] = sub_array
        #subfield[:] = sub_array 
        
        #Subtract background
        data_sub = raw_data - subfield
        return data_sub



    def create_image_from_raw(raw_data, background, norm, black, white, gamma, mask=None, threshold=10.0):

        # Normalize
        if norm is not None:
            raw_data = raw_data.astype(float)

            # Effektive Norm: mask<=threshold -> inf (drückt Pixel auf 0)
            if mask is not None:
                norm_eff = np.where(mask.astype(float) > float(threshold), norm.astype(float), np.inf)
                norm_eff = np.where(norm_eff == 0, np.inf, norm_eff)
            else:
                norm_eff = np.where(norm == 0, np.inf, norm.astype(float))

            raw_data = np.true_divide(raw_data, norm_eff, where=np.isfinite(norm_eff))
            np.absolute(raw_data, raw_data)
            data = np.clip(raw_data, 0, 1)
        else:
            data = raw_data  # falls keine Norm übergeben wurde, Rohdaten weiterreichen
        ...
        w = Worker((raw_data.shape[1], raw_data.shape[0]))
        img = w.makeImg(data=data)  # hier besser die geclippten Daten nutzen
       # Apply black and white points
        contrast = 1 / (white - black) if white - black > 0.01 else 100
        brightness = float(black)

        #scale from 0,1 to 0,255
        data = np.clip((255*contrast*(raw_data-brightness)),0,255).astype(np.uint8)
        # Helligkeit/Kontrast von [0,1] nach [0,255] abbilden
        contrast = 1 / (white - black) if white - black > 0.01 else 100
        brightness = float(black)
        data = np.clip(255*contrast*(data - brightness), 0, 255).astype(np.uint8)
        lut = THz.generateLUT()
        if gamma > 0:
            gamma_lut = np.clip(np.power(np.arange(256, dtype=float) / 255, gamma) * 255, 0, 255).astype(np.uint8)
            cv2.LUT(data,gamma_lut,data)


        temp = np.empty(data.shape+(3,),dtype=np.uint8)

        temp[:,:,0] = cv2.LUT(data,lut[0])
        temp[:,:,1] = cv2.LUT(data,lut[1])
        temp[:,:,2] = cv2.LUT(data,lut[2])
        cv2.cvtColor(temp,cv2.COLOR_HLS2BGR, temp)
        data = temp
        return data, img
    
    # def create_image_from_raw(raw_data, background, norm, black, white, gamma):
    #     """
    #     Creates an image from raw data with specified parameters.

    #     Args:
    #         raw_data:   Raw THz data
    #         background: Boolean
    #         norm:       Normalization factor
    #         black:      Black point value
    #         white:      White point value
    #         gamma:      Gamma correction value

    #     Returns:
    #         Processed image.
    #     """

    #     #Subtract bg noise
    #     if background:
    #         raw_data = THz.bg_subtraction(raw_data)
        

    #     # Normalize
    #     if norm is not None:
    #         raw_data = raw_data.astype(float)
    #         raw_data = np.true_divide(raw_data, norm)
    #         raw_data = np.absolute(raw_data,raw_data)
    #         data = np.clip(raw_data, 0,1)

        


    #     w = Worker((raw_data.shape[1], raw_data.shape[0]))
    #     img = w.makeImg(data=raw_data)
    #     # Apply black and white points
    #     contrast = 1 / (white - black) if white - black > 0.01 else 100
    #     brightness = float(black)

    #     #scale from 0,1 to 0,255
    #     data = np.clip((255*contrast*(raw_data-brightness)),0,255).astype(np.uint8)
    #     #data = np.subtract(255, data)        

    #     #Create LUT for color mapping
    #     lut = THz.generateLUT()
    #     #Convert to 8-bit image
    #     temp = np.empty(data.shape+(3,),dtype=np.uint8)


        # apply gamma correction
        # if gamma > 0:
        #     gamma_lut = np.clip(np.power(np.arange(256, dtype=float) / 255, gamma) * 255, 0, 255).astype(np.uint8)
        #     cv2.LUT(data,gamma_lut,data)
        
        # temp[:,:,0] = cv2.LUT(data,lut[0])
        # temp[:,:,1] = cv2.LUT(data,lut[1])
        # temp[:,:,2] = cv2.LUT(data,lut[2])
        # cv2.cvtColor(temp,cv2.COLOR_HLS2BGR, temp)
        # data = temp

    #     return data, img
    
    def generateLUT(mode="bw"):
        """ Generates look-up table to convert a value in 0-255 range into color.
        Parameters:
            mode --- either "fc" (false-color), or "bw" (blue-tinted black&white).
        Returns a tuple with three lookup tables for R, G, and B channels."""

        xxx = np.arange(256,dtype=np.uint8)

        temp = np.zeros((256,1,3),dtype=np.uint8)

        if mode == "bw":
            temp[:,0,0]=120
            temp[:,0,1]=xxx
            temp[:,0,2]=255
            cv2.cvtColor(temp,cv2.COLOR_HLS2BGR, temp)
        return (temp[:,0,0].copy(),temp[:,0,1].copy(),temp[:,0,2].copy())
    
    def set_norm_and_mask(self, norm: np.ndarray, mask: np.ndarray = None, threshold: float = None):
        """
        Übergib eine Norm-Matrix und optional eine SNR-Maske. Pixel mit mask<=threshold
        werden später auf 0 gedrückt (Norm dort = inf).
        """
        if threshold is not None:
            self.threshold = float(threshold)

        self.normSrc = norm.astype(float, copy=True)
        if mask is None:
            # "Alles gültig" falls keine Maske übergeben wird
            self.mask = np.full_like(self.normSrc, 1e99, dtype=float)
        else:
            self.mask = mask.astype(float, copy=True)

        self._prepare_norm()

        # Optional: auch an den Processor pushen, falls dessen Normalize() verwendet wird
        try:
            # Wenn dein Processor SetNorm(mask=...) kann, übergeben:
            self.p.SetNorm(self.normSrc, self.mask)
            if hasattr(self.p, "threshold"):
                self.p.threshold = self.threshold
        except Exception:
            pass

    def _prepare_norm(self):
        if self.normSrc is None:
            self.norm = None
            return
        m = self.mask if self.mask is not None else np.full_like(self.normSrc, 1e99, dtype=float)
        eff = np.where(m > self.threshold, self.normSrc, np.inf)
        # 0 in Norm vermeiden
        eff = np.where(eff == 0, np.inf, eff)
        self.norm = eff



# Main block
if __name__ == "__main__":

    # parms_thz = {
    # 'init': {
    #     'BELT_SPEED': 100,         # Beispielwert
    #     'FRAME_LENGTH': 2048,       # Beispielwert
    #     'ASPECT_RATIO': 1.0,       # Beispielwert
    #     'LIVESTREAM_VISIBLE': True,
    #     'RATE': 10                 # Beispielwert
    # },
    # 'n_images': 10                 # Beispielwert
    # }

    # # THz-Objekt erzeugen
    # thz = THz(parms_thz)

    # # Livestream starten
    # thz.start_stream()


    data_raw = np.load(r"IFI_git\Inline_Food_Inspection\sub_test.npy", allow_pickle=True)
    X_SIZE,Y_SIZE = data_raw.shape
    print(f"Image size: {X_SIZE}x{Y_SIZE}")
    black = 0.3
    white = 1.0
    gamma = 0.2

    MAX_VALUE = (1<<15)-1

    data, img = THz.create_image_from_raw(
        raw_data=data_raw,
        background=None,
        norm=np.ones((X_SIZE,Y_SIZE),dtype=float)*MAX_VALUE,
        black=black,
        white=white,
        gamma=gamma
    )
    plt.imsave("raw_image.png", data)
    cv2.imshow("Processed THz Image", data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


