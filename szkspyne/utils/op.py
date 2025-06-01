from torch.utils.data import Dataset
from pathlib import Path
from pydantic import BaseModel
from szkspyne.type.op import OpParams
from szkspyne.type import LineMixin
from PIL import Image


class OpData(BaseModel):
    op_params: OpParams
    image_path: str

    def __repr__(self):
        return f"OpData(op_params={self.op_params}, image_path={self.image_path})"
    
    def load_image(self):
        return Image.open(self.image_path)
    
    def to_vector(self):
        return self.op_params.to_vector()
    
    @staticmethod
    def from_directory(data_dir: Path) -> list['OpData']:
        op_data_list = []
        for json_fn in data_dir.glob('*.json'):
            json_fn = json_fn.resolve()
            image_fn = json_fn.with_suffix('.jpg')
            if not image_fn.exists():
                raise FileNotFoundError(f"Image file {image_fn} does not exist for {json_fn}")
            
            with open(json_fn, 'r') as f:
                op_params = OpParams.model_validate_json(f.read())
            op_data = OpData(op_params=op_params, image_path=str(image_fn))
            op_data_list.append(op_data)
        
        return op_data_list

class OpDataLine(BaseModel, LineMixin):
    content: OpData
    filename: str

class OpDataset(Dataset):
    def __init__(self, op_data_list: list[OpData], transform=None, image_only=False):
        self.data = op_data_list
        self.transform = transform
        self.image_only = image_only

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        op_data = self.data[idx]
        image = op_data.load_image()
        if self.image_only:
            if self.transform:
                image = self.transform(image)
            return image
        vector = op_data.to_vector()
        if self.transform:
            image, vector = self.transform(image, vector)
        return image, vector
    
    @staticmethod
    def from_directory(data_dir: Path):
        op_data_list = OpData.from_directory(data_dir)
        return OpDataset(op_data_list)