import os
from pathlib import Path
from PIL import Image
import pandas as pd
from ssim import SSIM
import numpy as np
import cv2
from image_similarity_measures.quality_metrics import fsim

class IFIAnalysis:
    """
    This class implements all methods required for the execution of analysis for Inline Food Inspection.
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (str or Path): Path to main folder.
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise ValueError(f"{root_dir!r} is not a valid path.")
        self.subdirs = [p for p in self.root_dir.iterdir() if p.is_dir()]

    def compute_all(self) -> pd.DataFrame:
        rows = []

        for subdir in self.subdirs:
            img_path = subdir / "thz.jpg"
            metadata = {'cookie': None, 
                        'dirt': None, 
                        'dirt_size': None, 
                        'paper': None}
            cw_value = fsim_score = chi2 = None

            # Extract meta data
            parts = subdir.name.split('_')
            if len(parts) >= 5:
                metadata['cookie']    = parts[0] + parts[1]
                metadata['dirt']      = parts[2]
                metadata['dirt_size'] = parts[3][:-2]
                metadata['paper']     = parts[4][:-1]

            if img_path.exists():
                try:
                    # Load image and crop
                    with Image.open(img_path) as img:
                        w, h = img.size
                        half_h = h // 2
                        top = img.crop((0, 0, w, half_h))
                        bot = img.crop((0, half_h, w, half_h + half_h))

                    # Ensure both halves have identical height
                    top_arr = np.array(top)
                    bottom_arr = np.array(bot)
                    min_h = min(top_arr.shape[0], bottom_arr.shape[0])
                    top = top.crop((0, 0, w, min_h))
                    bot = bot.crop((0, 0, w, min_h))

                    # Apply CW‑SSIM metric calculation
                    ssim = SSIM(top, gaussian_kernel_1d=None, size=(w, half_h))
                    cw_value = ssim.cw_ssim_value(bot)

                    # Apply FSIM metric calculation
                    top  = np.asarray(top)
                    bot  = np.asarray(bot)
                    # liefert float
                    fsim_score = fsim(org_img=top, pred_img=bot)

                    # Apply Chi‑Square distance metric calculation
                    top_gray = np.array(top.convert('L'))
                    bot_gray = np.array(bot.convert('L'))
                    h1 = cv2.calcHist([top_gray], [0], None, [256], [0,256]).flatten()
                    h2 = cv2.calcHist([bot_gray], [0], None, [256], [0,256]).flatten()
                    h1 /= (h1.sum() + 1e-10)
                    h2 /= (h2.sum() + 1e-10)
                    chi2 = 0.5 * np.sum((h1 - h2)**2 / (h1 + h2 + 1e-10))

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

            rows.append({**metadata,
                         'cw_ssim': cw_value,
                         'fsim': fsim_score,
                         'chi2': chi2})

        df = pd.DataFrame(rows, columns=['cookie', 
                                         'dirt', 
                                         'dirt_size', 
                                         'paper', 
                                         'cw_ssim', 
                                         'fsim', 
                                         'chi2'])
        return df


# Beispiel für die Nutzung:
if __name__ == "__main__":

    # Define path components
    current_path = os.getcwd()
    folder_name = "thz_processed_image_extracts"
    path_dir = os.path.join(current_path, folder_name)
    type_cookie = ['chocolate_wafer', 
                    'vanilla_crescent', 
                    'wafer_roll', 
                    'jelly_cookie']

    # Apply analysis operations for every cookie type
    for cookie in type_cookie:

        # Apply analysis
        batch = IFIAnalysis(os.path.join(path_dir, cookie))
        print(batch)
        df_results = batch.compute_all()

        # Export results
        df_results.to_csv(f"cw_ssim_results2_{cookie}.csv", index=False)
        print(df_results)
