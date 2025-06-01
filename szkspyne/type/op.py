from pydantic import BaseModel
from typing import ClassVar
import json

from szkspyne.type import LineMixin

class ScrewCountLr(BaseModel):
    left: dict[str, float]
    right: dict[str, float]

class OpParams(BaseModel):
    screw_counts: ScrewCountLr
    tilt_angles: list[float]
    version: str

    LABELS: ClassVar = ['T' + str(i) for i in range(1, 13)] + ['L' + str(i) for i in range(1, 5)]

    def to_vector(self):
        '''Convert OpParams to a vector representation.
        Vector format:
        [left_screw_count, right_screw_count, tilt_angle_1, tilt_angle_2, ...,]
        screw_counts contains counts T1 to T12 and L1 to L4
        '''
        left_counts = [self.screw_counts.left[label] for label in OpParams.LABELS]  
        right_counts = [self.screw_counts.right[label] for label in OpParams.LABELS]
        tilt_angles = self.tilt_angles
        return left_counts + right_counts + tilt_angles
    
    @staticmethod
    def from_vector(vector: list[float]):
        '''Convert a vector representation back to OpParams.
        Expects vector format:
        [left_screw_count, right_screw_count, tilt_angle_1, tilt_angle_2, ...]
        '''
        left_counts = vector[:len(OpParams.LABELS)]
        right_counts = vector[len(OpParams.LABELS):2*len(OpParams.LABELS)]
        tilt_angles = vector[2*len(OpParams.LABELS):]
        screw_counts = ScrewCountLr(left=dict(zip(OpParams.LABELS, left_counts)),
                                    right=dict(zip(OpParams.LABELS, right_counts)))
        return OpParams(screw_counts=screw_counts, tilt_angles=tilt_angles, version='from_vector')

class OpParamsLine(BaseModel, LineMixin):
    content: OpParams
    filename: str
