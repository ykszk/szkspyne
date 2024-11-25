from typing import Any, Dict,  Union

from pydantic import BaseModel


class MeasureResult(BaseModel):

    class Ok(BaseModel):
        Ok: Union[float, list[float]]

        def __init__(self, Ok: Union[float, list[float]]):
            super().__init__(Ok=Ok)

        def is_ok(self):
            return True
        
        def is_error(self):
            return False

    class Error(BaseModel):
        Err: Any

        def __init__(self, Err: Dict[str, Any]):
            super().__init__(Err=Err)

        def is_ok(self):
            return False
        
        def is_error(self):
            return True


    measurements: Dict[str, Union[Ok, Error]]
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
                measurements[k] = cls.Ok(ak.Ok - bk.Ok)
            elif isinstance(ak, cls.Error):
                measurements[k] = ak.model_copy()
            else:
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