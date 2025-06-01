from typing import List, Union
import json
from pathlib import Path

class LineMixin():
    '''Mixin class for lines that can be saved to and loaded from ndjson files.'''

    @classmethod
    def from_ndjson(cls, filename: Union[str, Path]) -> List:
        '''Load ndjson file and return a list of cls instances
        '''
        with open(filename) as f:
            lines = f.readlines()
        return [cls.model_validate_json(line) for line in lines]
    
    @classmethod
    def save_to_ndjson(cls, data: List['LineMixin'], filename: Union[str, Path], ensure_ascii=False):
        '''Save a list of cls instances to ndjson file
        '''
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item.model_dump(), ensure_ascii=False, separators=(',', ':')))
                f.write('\n')
