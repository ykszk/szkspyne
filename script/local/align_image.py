#%%
import torch
from pathlib import Path

# fixed_dir = Path('./devsubset/mismatch_jpg')
# moving_dir = Path('./devsubset/imgs_for_annotation')
# outdir = Path('./devsubset/aligned')

fixed_dir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/misaligned')
moving_dir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/imgs_for_annotation')
outdir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/misaligned_aligned')


# fixed_dir = Path('./mismatch_jpg')
# moving_dir = Path('./imgs_for_annotation')
# outdir = Path('./aligned')

#%%
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional.clustering import mutual_info_score
from tqdm import tqdm
import scipy.ndimage as ndi
import numpy as np
import cv2
# https://raw.githubusercontent.com/xzluo97/mutual-information-registration/ec0c05915e41c4222e184c2feb79cc1e7e487631/MI.py
import mi_reg

class RigidRegistration(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.zeros(2,1), requires_grad=True)
        # self.affine = torch.tensor([[[1, 0, self.t[0]], [0, 1, self.t[1]]]], dtype=torch.float32)
        # self.affine = torch.cat([torch.eye(2), self.t], dim=1)[None]
        self.mi = mi_reg.MI(2, num_bins=128, normalize=True)
        self.losses = []
    
    def get_affine(self):
        return torch.cat([torch.eye(2).to(self.t.device), self.t], dim=1)[None]
    
    def align_by_center_of_mass(self, moving: np.ndarray, fixed: np.ndarray):
        moving = moving.squeeze()
        fixed = fixed.squeeze()
        moving_com = ndi.measurements.center_of_mass(moving)
        fixed_com = ndi.measurements.center_of_mass(fixed)
        real_t = torch.tensor(moving_com) - torch.tensor(fixed_com)
        scaled_t = real_t / torch.tensor(moving.shape)
        self.t = nn.Parameter(torch.tensor([[scaled_t[1]], [scaled_t[0]]],  dtype=torch.float32, requires_grad=True))
        # print('self.t',self.t)

    def align_by_phase_correlation(self, moving: np.ndarray, fixed: np.ndarray):
        moving = moving.squeeze().astype(np.float32)
        fixed = fixed.squeeze().astype(np.float32)
        moving = pad_moving(fixed, moving)
        (x,y), _ = cv2.phaseCorrelate(moving, fixed)
        real_t = torch.tensor([-y, -x])
        # print('real_t', real_t)
        scaled_t = real_t / torch.tensor(moving.shape)
        self.t = nn.Parameter(torch.tensor([[scaled_t[1]], [scaled_t[0]]],  dtype=torch.float32, requires_grad=True))


    def transform(self, moving):
        affine=  self.get_affine()
        # print('affine', affine)
        grid = F.affine_grid(affine, moving.shape)
        return F.grid_sample(moving, grid, padding_mode='zeros', align_corners=True)
    
    def fit(self, moving, fixed, n_iter, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        moving = moving.clone().detach().requires_grad_(True)
        progress_bar = tqdm(range(n_iter), total=n_iter)
        for _ in progress_bar:
            optimizer.zero_grad()
            moved = self.transform(moving)
            # affine = torch.tensor([[[1, 0, self.t[0]], [0, 1, self.t[1]]]], dtype=torch.float32)
            # grid = F.affine_grid(affine, moving.shape)
            # moved = F.grid_sample(moving, grid)
            # self.t.retain_grad()
            # loss = -mutual_info_score(moved.ravel(), fixed.ravel())
            # print('minmax', moved.min(), moved.max())
            loss = -self.mi.mi(moved, fixed)
            # loss = nn.MSELoss()(moved, fixed)
            loss.backward()
            # print(self.t, loss, self.t.grad)
            progress_bar.desc = f"Loss: {loss.item()}"
            optimizer.step()
            self.losses.append(loss.item())


import numpy as np
from scipy import signal
from PIL import Image
# from torchreg import AffineRegistration
import matplotlib.pyplot as plt
# from torchmetrics.clustering import MutualInfoScore

def pad_moving(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    shape_diff = np.array(fixed.shape) - np.array(moving.shape)
    pad_width = np.maximum(shape_diff, 0)
    pad_width = [(0, w) for w in pad_width]
    moving = np.pad(moving, pad_width, mode='constant')
    return moving

def diff2img(diff) -> Image.Image:
    diff = diff.squeeze()
    diff = diff - diff.min()
    denom = diff.max() - diff.min()
    if denom == 0:
        return Image.fromarray(np.zeros_like(diff))
    diff = diff / denom * 255
    diff = diff.astype(np.uint8)
    return Image.fromarray(diff)

def dissimilarity(fixed: torch.Tensor, moved: torch.Tensor) -> torch.Tensor:
    return - mutual_info_score(fixed.ravel(), moved.ravel())

def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    tmp = signal.decimate(image, factor, axis=0)
    return np.ascontiguousarray(signal.decimate(tmp, factor, axis=1))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for fixed_path in sorted(fixed_dir.glob('*.jpg')):
    print(f"Processing {fixed_path}")
    moving_path = moving_dir / fixed_path.name
    if not moving_path.exists():
        print(f"Moving file not found: {moving_path}")
        continue
    original_fixed = np.array(Image.open(fixed_path))
    original_moving = np.array(Image.open(moving_path))
    padded_moving = pad_moving(original_fixed, original_moving)
    regs = []
    reg = RigidRegistration()
    reg.align_by_phase_correlation(original_moving, original_fixed)
    regs.append(reg)
    reg = RigidRegistration()
    reg.align_by_center_of_mass(original_moving, original_fixed)
    regs.append(reg)

    def register(reg, n_iter, sample_rate, lr):
        reg.n_iter = n_iter
        fixed = downsample(original_fixed, sample_rate)[np.newaxis, np.newaxis]
        moving = downsample(np.array(padded_moving), sample_rate)[np.newaxis, np.newaxis]
        fixed = torch.tensor(fixed).float().to(device)
        moving = torch.tensor(moving).float().to(device)
        reg.to(device)
        reg.fit(moving, fixed, n_iter, lr)


    for reg in regs:
        for (sample_rate, n_iter, lr) in [(10, 200, 0.03)]:
            register(reg, n_iter, sample_rate, lr)

    regs.sort(key=lambda x: x.losses[-1])
    if regs[0].losses[-1] > regs[1].losses[-1]:
        print('Use center of mass')
        reg = regs[1]
    else:
        print('Use phase correlation')
        reg = regs[0]

    for (sample_rate, n_iter, lr) in [(5, 100, 0.002), (2, 100, 0.001)]:
        register(reg, n_iter, sample_rate, lr)
    reg.to('cpu')
    moved = reg.transform(torch.tensor(padded_moving[np.newaxis, np.newaxis], dtype=torch.float32)).detach().numpy().squeeze()

    moved_diff = original_fixed - moved
    moved_diff = diff2img(moved_diff)
    moved_diff.save(outdir / fixed_path.name)
    original_diff = original_fixed - padded_moving
    original_diff = diff2img(original_diff)
    original_diff.save(outdir / f"{fixed_path.stem}_original.jpg")

    plt.plot(reg.losses)
    plt.savefig(outdir / f"loss_{fixed_path.stem}.png")
    plt.clf()

    params = reg.t.detach().numpy()
    np.save(outdir / f"params_{fixed_path.stem}.npy", params)

    # break