import argparse
from pathlib import Path
import pydicom
import pydicom.uid
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import namedtuple
from typing import Dict
import warnings

warnings.filterwarnings("error")


CropParams = namedtuple("CropParams", ["y_start", "y_end", "x_start", "x_end"])


def resize_dicom(dicom_path: str, output: str, crop_params: Dict[str, CropParams]):
    import pydicom.config

    pydicom.config.RAISE = 2
    # import warnings
    # warnings.filterwarnings("error")
    # if Path(dicom_path).name!='OSC00214_20080507_8.dcm':
    #     return
    # print("Processing", dicom_path)
    dcm = pydicom.dcmread(dicom_path)  # , stop_before_pixels=True)
    # try:
    #     if dcm.BitsAllocated != 16:
    #         raise ValueError(f"Invalid bits_allocated in {dicom_path}: {dcm.bits_allocated}")
    # except Exception as e:
    #     print(f"Error in {dicom_path}: {e}")
    # return
    try:
        arr = dcm.pixel_array
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            # arr = 2**dcm.BitsStored - 1 - arr
            arr = arr.max() - arr
        # normalize to 12bit
        if arr.dtype not in ["uint16", "int16"]:
            raise ValueError(f"Invalid dtype in {dicom_path}: {arr.dtype}")
        arr = arr.astype("float64")
        arr = arr - arr.min()
        arr = (arr / arr.max() * 8191).astype("uint16")
        dcm.BitsStored = 13
        dcm.BitsAllocated = 16
        dcm.HighBit = 12
        try:
            del dcm[0x0040, 0x0260]  # Performed Protocol Code Sequence Attribute
        except:
            pass

        result = arr[::2, ::2]
        cp = crop_params.get(Path(dicom_path).stem, None)
        # print(cp)
        if cp is not None:
            result = result[cp.y_start : cp.y_end, cp.x_start : cp.x_end]
        else:
            if len(crop_params) > 0:
                print(f"No crop params for {dicom_path}")
                return
            pass
        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dcm.PhotometricInterpretation = "MONOCHROME2"
        dcm.SoftwareVersions = "pydicom suzuki"
        dcm.PixelRepresentation = 0
        try:
            dcm.PixelSpacing = "{}\\{}".format(
                dcm.PixelSpacing[0] * 2, dcm.PixelSpacing[1] * 2
            )
        except Exception as e:
            print(e)
        try:
            dcm.ImagerPixelSpacing = "{}\\{}".format(
                dcm.ImagerPixelSpacing[0] * 2, dcm.ImagerPixelSpacing[1] * 2
            )
        except Exception as e:
            print(e)
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
    parser.add_argument(
        "--crop_param", help="Crop parameter csv", type=Path, default=None
    )
    parser.add_argument("-j", "--jobs", help="Number of jobs", type=int, default=1)
    args = parser.parse_args()

    if args.crop_param is not None:
        import pandas as pd

        df = pd.read_csv(args.crop_param)
        # crop_params = {row['filename']: row[['y_start', 'y_end', 'x_start', 'x_end']].values for _, row in df.iterrows()}
        crop_params = {
            row["filename"]: CropParams(
                *row[["y_start", "y_end", "x_start", "x_end"]].values
            )
            for _, row in df.iterrows()
        }
    else:
        crop_params = {}

    filenames = sorted(args.input.glob("*.dcm"))
    Parallel(n_jobs=args.jobs)(
        delayed(resize_dicom)(dcm_path, args.output / dcm_path.name, crop_params)
        for dcm_path in tqdm(filenames)
    )
    # for dcm_path in tqdm(filenames):
    #     output = args.output / dcm_path.name
    #     resize_dicom(dcm_path, output, crop_params)


if __name__ == "__main__":
    main()
