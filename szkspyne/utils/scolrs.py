from typing import Annotated, Any, Dict, Optional, Union

from pydantic import BaseModel, Discriminator, Tag
import pandas as pd


class MeasureResult(BaseModel):

    class Ok(BaseModel):
        Ok: float

        def value(self):
            return self.Ok

    class Error(BaseModel):
        Err: Any

        def value(self):
            return self.Err

    @staticmethod
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
            else:
                assert isinstance(bk, cls.Error)
                measurements[k] = bk.model_copy()
        return MeasureResult(measurements=measurements, unit_of_length=a.unit_of_length)

    def difference(self, other: 'MeasureResult') -> 'MeasureResult':
        '''
        Calculate the differences (self - other) between two MeasureResult objects.
        '''
        return MeasureResult.differences(self, other)


class MeasureResultLine(BaseModel):
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

        return pd.DataFrame(rows)
