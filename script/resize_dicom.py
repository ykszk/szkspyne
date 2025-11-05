import argparse
from pathlib import Path
import pydicom
import pydicom.uid
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from collections import namedtuple
from typing import Dict
import warnings

warnings.filterwarnings("error")


def resize_dicom(dicom_path: str, output: str):
    import pydicom.config

    pydicom.config.RAISE = 2
    dcm = pydicom.dcmread(dicom_path)  # , stop_before_pixels=True)
    try:
        arr = dcm.pixel_array

        # resize by averaging 2x2 blocks
        height, width = arr.shape
        new_height = (height // 2) * 2
        new_width = (width // 2) * 2
        trimmed = arr[:new_height, :new_width]
        result = (trimmed.reshape(trimmed.shape[0] // 2, 2, trimmed.shape[1] // 2, 2).mean(axis=(1, 3))).astype(trimmed.dtype)
        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dcm.SoftwareVersions = "resize_dicom"
        dcm.PixelData = result.tobytes()
        dcm.Rows, dcm.Columns = result.shape
        compress = True
        if compress:
            dcm.compress(pydicom.uid.RLELossless)
        dcm.save_as(output)
    except Exception as e:
        print(f"Error in {dicom_path}: {e}")
    except:
        print(f"Error in {dicom_path}")


def main():
    parser = argparse.ArgumentParser(description="Resize DICOM files")
    parser.add_argument("input", help="Input directory", type=Path)
    parser.add_argument("output", help="Output directory", type=Path)
    parser.add_argument("-j", "--jobs", help="Number of jobs", type=int, default=-1)
    args = parser.parse_args()


    filenames = sorted(args.input.glob("*.dcm"))
    Parallel(n_jobs=args.jobs)(
        delayed(resize_dicom)(dcm_path, args.output / dcm_path.name)
        for dcm_path in tqdm(filenames)
    )


if __name__ == "__main__":
    main()
