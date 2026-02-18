from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Discriminator, Tag

from szkspyne.type import LineMixin


def get_discriminator_value(v: Any) -> Optional[str]:
    if isinstance(v, dict):
        if 'Ok' in v:
            return 'Ok'
        elif 'Err' in v:
            return 'Err'
        else:
            return None

    if hasattr(v, 'Ok'):
        return 'Ok'
    elif hasattr(v, 'Err'):
        return 'Err'
    else:
        return None


class MeasureResult(BaseModel):

    class Ok(BaseModel):
        Ok: float

        def value(self):
            return self.Ok

    class Error(BaseModel):
        Err: Any

        def value(self):
            return self.Err

    measurements: Dict[
        str,
        Annotated[
            Union[Annotated[Ok, Tag('Ok')], Annotated[Error, Tag('Err')]], Discriminator(get_discriminator_value)
        ],
    ]
    unit_of_length: str

    def ok_measurements(self):
        return {k: v for k, v in self.measurements.items() if isinstance(v, self.Ok)}

    def error_measurements(self):
        return {k: v for k, v in self.measurements.items() if isinstance(v, self.Error)}

    @classmethod
    def differences(cls, a: 'MeasureResult', b: 'MeasureResult') -> 'MeasureResult':
        '''
        Calculate the differences (a - b) between two MeasureResult objects.
        '''
        if a.unit_of_length != b.unit_of_length:
            raise ValueError('Unit of length is different')
        measurements = {}
        for k in a.measurements.keys():
            ak = a.measurements.get(k)
            bk = b.measurements.get(k)
            if isinstance(ak, cls.Ok) and isinstance(bk, cls.Ok):
                measurements[k] = cls.Ok(Ok=ak.Ok - bk.Ok)
                pass
            elif isinstance(ak, cls.Error):
                measurements[k] = ak.model_copy()
            elif isinstance(bk, cls.Error):
                measurements[k] = bk.model_copy()
            else:
                # raise ValueError(f'Measurement {k} is missing in one of the MeasureResult objects')
                measurements[k] = cls.Error(Err='Measurement is missing in one of the MeasureResult objects')
        return MeasureResult(measurements=measurements, unit_of_length=a.unit_of_length)

    def difference(self, other: 'MeasureResult') -> 'MeasureResult':
        '''
        Calculate the differences (self - other) between two MeasureResult objects.
        '''
        return MeasureResult.differences(self, other)


class MeasureResultLine(BaseModel, LineMixin):
    filename: str
    content: MeasureResult

    @classmethod
    def to_dataframe(cls, lines: list['MeasureResultLine']) -> pd.DataFrame:
        '''
        Convert a list of MeasureResultLine objects to a pandas DataFrame.
        '''
        rows = []
        for line in lines:
            row: Dict[str, Any] = {'filename': line.filename, 'unit_of_length': line.content.unit_of_length}
            row.update({k: v.value() for k, v in line.content.measurements.items()})
            rows.append(row)

        return pd.DataFrame(rows).set_index('filename')


class MeasureListResult(BaseModel):

    class Ok(BaseModel):
        Ok: List[float]

        def value(self):
            return self.Ok

    class Error(BaseModel):
        Err: Any

        def value(self):
            return self.Err

    measurements: Dict[
        str,
        Annotated[
            Union[Annotated[Ok, Tag('Ok')], Annotated[Error, Tag('Err')]],
            Discriminator(get_discriminator_value),
        ],
    ]
    unit_of_length: str

    def ok_measurements(self):
        return {k: v for k, v in self.measurements.items() if isinstance(v, self.Ok)}

    def error_measurements(self):
        return {k: v for k, v in self.measurements.items() if isinstance(v, self.Error)}

    @classmethod
    def differences(cls, a: 'MeasureListResult', b: 'MeasureListResult') -> 'MeasureListResult':
        '''
        Calculate the differences (a - b) between two MeasureListResult objects.
        '''
        if a.unit_of_length != b.unit_of_length:
            raise ValueError('Unit of length is different')
        measurements = {}
        for k in a.measurements.keys():
            ak = a.measurements.get(k)
            bk = b.measurements.get(k)
            if isinstance(ak, cls.Ok) and isinstance(bk, cls.Ok):
                measurements[k] = cls.Ok(Ok=[x - y for x, y in zip(ak.Ok, bk.Ok)])
                pass
            elif isinstance(ak, cls.Error):
                measurements[k] = ak.model_copy()
            else:
                assert isinstance(bk, cls.Error)
                measurements[k] = bk.model_copy()
        return MeasureListResult(measurements=measurements, unit_of_length=a.unit_of_length)

    def difference(self, other: 'MeasureListResult') -> 'MeasureListResult':
        '''
        Calculate the differences (self - other) between two MeasureListResult objects.
        '''
        return MeasureListResult.differences(self, other)

    def flatten(self) -> MeasureResult:
        '''
        Flatten the MeasureListResult object to MeasureResult object.
        If there are multiple values in the Ok field, suffixes are added to the keys.
        '''
        measurements = {}
        for k, v in self.measurements.items():
            if isinstance(v, self.Ok):
                if len(v.Ok) == 1:
                    measurements[k] = MeasureResult.Ok(Ok=v.Ok[0])
                else:
                    for i, x in enumerate(v.Ok, 1):
                        measurements[f'{k}_{i}'] = MeasureResult.Ok(Ok=x)
            else:
                measurements[k] = MeasureResult.Error(Err=v.Err)
        return MeasureResult(measurements=measurements, unit_of_length=self.unit_of_length)


class MeasureListResultLine(BaseModel, LineMixin):
    filename: str
    content: MeasureListResult

    @classmethod
    def to_dataframe(cls, lines: list['MeasureListResultLine']) -> pd.DataFrame:
        '''
        Convert a list of MeasureListResultLine objects to a pandas DataFrame.
        '''
        rows = []
        for line in lines:
            row: Dict[str, Any] = {'filename': line.filename, 'unit_of_length': line.content.unit_of_length}
            row.update({k: v.value() for k, v in line.content.measurements.items()})
            rows.append(row)

        return pd.DataFrame(rows).set_index('filename')


class ImageMetadata(BaseModel):
    path: str
    height: int
    width: int
    spacing_xy: Tuple[float, float]
    unit: str


class CoronalPoints(BaseModel):
    spine: List[List[List[float]]]
    clavicle: List[List[float]]
    shoulder: List[List[float]]
    pelvis: List[List[float]]
    femoral_head: List[List[float]]
    image_metadata: ImageMetadata

class CoronalPointsLine(BaseModel, LineMixin):
    filename: str
    content: CoronalPoints

class SagittalPoints(BaseModel):
    spine: List[List[List[float]]]
    femoral_head: List[List[float]]
    image_metadata: ImageMetadata

class SagittalPointsLine(BaseModel, LineMixin):
    filename: str
    content: SagittalPoints


class SpinalPoints(BaseModel):
    spine: List[List[List[float]]]
    femoral_head: List[List[float]]
    image_metadata: ImageMetadata

class SpinalPointsLine(BaseModel, LineMixin):
    filename: str
    content: SpinalPoints


import argparse
import sys


def main():
    parser = argparse.ArgumentParser("Convert ndjson to csv")
    parser.add_argument('ndjson', type=str, help='ndjson file to load')
    parser.add_argument('csv', type=str, help='csv file to save')
    # remove "unit_of_length"
    parser.add_argument('--no-unit', action='store_true', help='Remove unit_of_length column in output csv')
    args = parser.parse_args()

    with open(args.ndjson, 'r') as f:
        lines = f.readlines()
    lines = [MeasureResultLine.model_validate_json(line) for line in lines]

    df = MeasureResultLine.to_dataframe(lines)
    if args.no_unit:
        df.drop(columns=['unit_of_length'], inplace=True)
    df.to_csv(args.csv, index=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
