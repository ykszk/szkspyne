import argparse
import itertools
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pydicom
import tqdm
from joblib import Parallel, delayed
from logzero import logger


meta_list = [
    'StudyInstanceUID',
    'PatientID',
    # 'AccessionNumber',
    # 'InstitutionName',
    'Modality',
    'ImageType',
    'Rows',
    'Columns',
    'StudyDate',
    'SeriesDate',
    'SeriesTime',
    'AcquisitionNumber',
    # 'InstanceCreationTime',
    'ContentDate',
    'ContentTime',
    'ProtocolName',
    'SeriesDescription',
    'InstanceNumber' 'SeriesNumber',
    'StudyDescription',
    'SeriesDescription',
    'PixelSpacing',
    'ImagerPixelSpacing',
    'Manufacturer',
    'ManufacturerModelName',
    'StudyInstanceUID',
    'SeriesInstanceUID',
    'StudyTime',
]


def extract_metadata(zip_filename):
    with zipfile.ZipFile(zip_filename) as zf:
        ms = []
        for filename in zf.namelist():
            blob = zf.read(filename)
            with BytesIO(blob) as bio:
                ds = pydicom.dcmread(bio, stop_before_pixels=True)
            meta = {'filename': f'{zip_filename.name}/{filename}'}
            meta.update({m: getattr(ds, m, '') for m in meta_list})

            ms.append(meta)
        return ms


def main():
    parser = argparse.ArgumentParser(description='Collect metadata from zipped dicom files.')
    parser.add_argument('input', help='Input zipped dicom directory', metavar='<input>')
    parser.add_argument('output', help='Output metadata csv', metavar='<output>')

    args = parser.parse_args()

    rows = Parallel(n_jobs=8)(
        delayed(extract_metadata)(arg) for arg in tqdm.tqdm(sorted(Path(args.input).glob('*.zip')))
    )
    if rows is not None:
        rows = itertools.chain.from_iterable(rows)
    logger.info('Saving dataframe')
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, encoding='utf8')


if __name__ == '__main__':
    sys.exit(main())
