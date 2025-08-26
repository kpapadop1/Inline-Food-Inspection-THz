import time
import threading
import sys
import os
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

    def start_stream(self):
        """
        Starts the processor stream and optionally the console thread. Must be called before execute_measurement_simple().

        Args:

        Returns:

        Raises:
        """

        if not self.livestream_visible:
            threading.Thread(target=self._console_input_loop, daemon=True).start()
        self.p.start(self.on_result, self.on_error)

    def terasense_stop(self, source):
        """
        Stops the processor stream and closes windows.

        Args:
            source (TeraSense Processor):      TeraSense Processor object

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
    
    def create_image_from_raw(raw_data, background, norm, black, white, gamma):
        """
        Creates an image from raw data with specified parameters.

        Args:
            raw_data (Numpy arra):      Raw THz data
            background (Numpy arra):    Background image
            norm (Numpy arra):          Normalization factor
            black (float):              Black point value
            white (float):              White point value
            gamma (float):              Gamma correction value

        Returns:
            Processed image.

        Raises:
        """
        # Subtract bg noise
        if background is not None:
            raw_data = raw_data - background
        
        # Normalize
        if norm is not None:
            raw_data = np.true_divide(raw_data, norm)
            raw_data = np.absolute(raw_data,raw_data)
            data = np.clip(raw_data, 0,1)

        w = Worker((raw_data.shape[1], raw_data.shape[0]))
        img = w.makeImg(data=raw_data)

        # Apply black and white points
        contrast = 1 / (white - black) if white - black > 0.01 else 100
        brightness = float(black)

        # Scale from 0,1 to 0,255
        data = np.clip((255*contrast*(raw_data-brightness)),0,255).astype(np.uint8)    

        # Create LUT for color mapping
        lut = THz.generateLUT()
        # Convert to 8-bit image
        temp = np.empty(data.shape+(3,),dtype=np.uint8)

        # apply gamma correction
        if gamma > 0:
            gamma_lut = np.clip(np.power(np.arange(256, dtype=float) / 255, gamma) * 255, 0, 255).astype(np.uint8)
            cv2.LUT(data,gamma_lut,data)
        
        temp[:,:,0] = cv2.LUT(data,lut[0])
        temp[:,:,1] = cv2.LUT(data,lut[1])
        temp[:,:,2] = cv2.LUT(data,lut[2])
        cv2.cvtColor(temp,cv2.COLOR_HLS2BGR, temp)
        data = temp

        return data, img
    
    def generateLUT(mode="bw"):
        """ Generates look-up table to convert a value in 0-255 range into color.

        Args:
            mode (str): Either "fc" (false-color), or "bw" (blue-tinted black&white).

        Returns:
            Tuple with three lookup tables for R, G, and B channels."""

        xxx = np.arange(256,dtype=np.uint8)

        temp = np.zeros((256,1,3),dtype=np.uint8)

        if mode == "bw":
            temp[:,0,0]=120
            temp[:,0,1]=xxx
            temp[:,0,2]=255
            cv2.cvtColor(temp,cv2.COLOR_HLS2BGR, temp)
        return (temp[:,0,0].copy(),temp[:,0,1].copy(),temp[:,0,2].copy())
    

# Main block
if __name__ == "__main__":

    # Define constants
    NAME_FILE = r"ifi_thz_raw_2025_07_08-10_22_50.pkl"
    BLACK = 0.3
    WHITE = 1.0
    GAMMA = 0.2
    MAX_VALUE = (1<<15)-1

    # Set path
    current_path = os.getcwd()
    data_raw = np.load(os.path.join(current_path, NAME_FILE), allow_pickle=True)
    X_SIZE, Y_SIZE = data_raw.shape

    # Apply operations to create images and show them
    data, img = THz.create_image_from_raw(
        raw_data=data_raw,
        background=None,
        norm=np.ones((X_SIZE, Y_SIZE),dtype=float)*MAX_VALUE,
        black=BLACK,
        white=WHITE,
        gamma=GAMMA)
    cv2.imshow("Processed THz Image", data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

