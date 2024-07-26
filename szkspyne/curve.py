# %%
from enum import Enum
from typing import List, Optional, Tuple, Dict
import argparse

import numpy as np
from pydantic import BaseModel, Field
from utils import measure
from utils.measure import VertebraDiscIndex


class Curve(BaseModel):
    sup: int
    inf: int


CurveAngle = Tuple[Curve, float]
CurveAngleApex = Tuple[Curve, float, VertebraDiscIndex]


class CurveSet(BaseModel):
    PT: Optional[Tuple[Curve, float]]
    MT: Optional[Tuple[Curve, float]]
    TLL: Optional[Tuple[Curve, float]]

    @staticmethod
    def empty() -> 'CurveSet':
        return CurveSet(PT=None, MT=None, TLL=None)


class ApexSet(BaseModel):
    PT: Optional[VertebraDiscIndex]
    MT: Optional[VertebraDiscIndex]
    TLL: Optional[VertebraDiscIndex]

    @staticmethod
    def empty() -> 'ApexSet':
        return ApexSet(PT=None, MT=None, TLL=None)


class MajorCurve(Enum):
    MT = 'MT'
    TLL = 'TLL'

class PyCurve(BaseModel):
    name: str
    indices: Dict[str, Optional[List[int]]]
    apices: Dict[str, Optional[int]]
    angles: Dict[str, Optional[float]]

class PyCurveInfo(BaseModel):
    curve_indices: List[PyCurve]

class PyCurveLine(BaseModel):
    filename: str
    content: PyCurveInfo

class CurveInfo(BaseModel):
    curves: CurveSet
    apices: ApexSet
    major_curve: Optional[MajorCurve]

    # def alt_apex(self) -> Optional[VertebraDiscIndex]:
    #     '''
    #     Alternative apex if current apex is on the boundary
    #     '''
    #     if self.major_curve is None:
    #         return None
    #     if self.major_curve == MajorCurve.MT:
    #         if self.apices in [VertebraDiscIndex.DT4T5, VertebraDiscIndex.T5]:
    #             return VertebraDiscIndex.T6
    #         return VertebraDiscIndex.T6
    #     else:
    #         return VertebraDiscIndex.T6
    
    def to_py_curve(self, name) -> PyCurve:
        indices: Dict[str, Optional[List[int]]] = {k: None for k in self.curves.model_dump().keys()}
        angles: Dict[str, Optional[float]] = {k: None for k in self.apices.model_dump().keys()}
        for k, v in self.curves.model_dump().items():
            if v is not None:
                indices[k] = [v[0]['sup'], v[0]['inf']]
                angles[k] = v[1]
        apices: Dict[str, Optional[int]] = {k: None for k in self.apices.model_dump().keys()}
        for k, v in self.apices.model_dump().items():
            if v is not None:
                apices[k] = v.value

        return PyCurve(name=name, indices=indices, apices=apices, angles=angles)

AllCurves = List[CurveAngleApex]


def get_curve(all_curves: AllCurves, i: int) -> Tuple[Optional[CurveAngle], Optional[VertebraDiscIndex]]:
    if 0 <= i and i < len(all_curves):
        c = all_curves[i]
        return ((c[0], c[1]), c[2])
    else:
        return (None, None)

class CurveInfoAll(BaseModel):
    curves: CurveSet
    apices: ApexSet
    all_curves: AllCurves
    major_curve: Optional[MajorCurve]

    def second_info(self) -> Optional[CurveInfo]:
        angles = [c[1] for c in self.all_curves]
        i_largest: int = np.argmax(np.array(angles))
        largest_curve = self.all_curves[i_largest]
        apex = largest_curve[2]
        apices = ApexSet.empty()
        curves = CurveSet.empty()

        # PT and MT border
        if apex in [VertebraDiscIndex.DT4T5, VertebraDiscIndex.T5]:
            major_curve = MajorCurve.MT
            apices.MT = VertebraDiscIndex.T6
            curves.MT = (largest_curve[0], largest_curve[1])
            curves.TLL, apices.TLL = get_curve(self.all_curves, i_largest + 1)
            curves.PT, apices.PT = get_curve(self.all_curves, i_largest - 1)
            return CurveInfo(curves=curves, apices=apices, major_curve=major_curve)
        if apex in [VertebraDiscIndex.DT5T6, VertebraDiscIndex.T6]:
            major_curve = MajorCurve.MT  # PT is can't be the major curve
            apices.PT = VertebraDiscIndex.T5
            curves.PT = (largest_curve[0], largest_curve[1])
            curves.MT, apices.MT = get_curve(self.all_curves, i_largest + 1)
            curves.TLL, apices.TLL = get_curve(self.all_curves, i_largest + 2)
            return CurveInfo(curves=curves, apices=apices, major_curve=major_curve)

        # MT and TLL border
        if apex in [VertebraDiscIndex.T11, VertebraDiscIndex.DT11T12]:
            major_curve = MajorCurve.TLL
            apices.TLL = VertebraDiscIndex.T12
            curves.TLL = (largest_curve[0], largest_curve[1])
            curves.MT, apices.MT = get_curve(self.all_curves, i_largest - 1)
            curves.PT, apices.PT = get_curve(self.all_curves, i_largest - 2)
            return CurveInfo(curves=curves, apices=apices, major_curve=major_curve)
        if apex in [VertebraDiscIndex.T12, VertebraDiscIndex.DT12L1]:
            major_curve = MajorCurve.MT
            apices.MT = VertebraDiscIndex.DT11T12
            curves.MT = (largest_curve[0], largest_curve[1])
            curves.TLL, apices.TLL = get_curve(self.all_curves, i_largest + 1)
            curves.PT, apices.PT = get_curve(self.all_curves, i_largest - 1)
            return CurveInfo(curves=curves, apices=apices, major_curve=major_curve)
        return None

    def to_curve_info(self) -> CurveInfo:
        return CurveInfo(curves=self.curves, apices=self.apices, major_curve=self.major_curve)

class CurveInfoAllLine(BaseModel):
    filename: str
    content: CurveInfoAll


# class PyCurve(BaseModel):
#     curve_indices: List[Optional[CurveAngle]]


# test_input = '''
# {"curves":{"PT":[{"sup":1,"inf":4},-2.3623974],"MT":[{"sup":4,"inf":11},37.24298],"TLL":[{"sup":11,"inf":14},-16.772045]},"apices":{"PT":5,"MT":16,"TLL":25},"major_curve":"MT","all_curves":[[{"sup":11,"inf":14},-16.772045,25],[{"sup":14,"inf":16},-8.394589,29],[{"sup":4,"inf":11},37.24298,16],[{"sup":1,"inf":4},-2.3623974,5]]}
# '''

# info = CurveInfoAll.model_validate_json(test_input, strict=True)
# primary_info = info.to_curve_info()
# curve_dict = {'Primary': primary_info}
# second_info = info.second_info()
# if second_info:
#     curve_dict['Secondary'] = second_info


def main():
    import sys
    parser = argparse.ArgumentParser(description='Insert secondary curve info if possible.')
    parser.add_argument('input', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input ndjson filename')


    args = parser.parse_args()

    all_data = [CurveInfoAllLine.model_validate_json(line) for line in args.input]
    for data in all_data:
        content  = data.content
        curve_indices = [content.to_curve_info().to_py_curve('Primary')]
        second_info = content.second_info()
        if second_info:
            curve_indices.append(second_info.to_py_curve('Secondary'))
        print(PyCurveLine(filename=data.filename, content=PyCurveInfo(curve_indices=curve_indices)).model_dump_json())            

if __name__ == '__main__':
    main()