#%%
import torch
from pathlib import Path
import argparse
# fixed_dir = Path('./devsubset/mismatch_jpg')
# moving_dir = Path('./devsubset/imgs_for_annotation')
# outdir = Path('./devsubset/aligned')

fixed_dir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/misaligned')
moving_dir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/imgs_for_annotation')
outdir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/misaligned_aligned')


# fixed_dir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/mismatch_jpg')
# moving_dir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/imgs_for_annotation')
# outdir = Path('/poseidon/users/suzuki/projects/scoliosis/data/dicom/aligned')

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
        moving_com = ndi.center_of_mass(moving)
        fixed_com = ndi.center_of_mass(fixed)
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
        grid = F.affine_grid(affine, moving.shape, align_corners=False)
        return F.grid_sample(moving, grid, padding_mode='zeros', align_corners=False)
    
    def fit(self, moving, fixed, n_iter, lr):
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
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
import gc
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

# device and skip option using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--skip_existing', action='store_true')
parser.add_argument('--jobs', type=int, default=1)
args = parser.parse_args()

device = args.device
skip_existing = True
fixed_paths = sorted(fixed_dir.glob('*.jpg'))
# for fixed_path in fixed_paths:
def run(fixed_path: Path, job_id: int):
    torch.cuda.empty_cache()
    if device=='cuda':
        # randomly set current device
        torch.cuda.set_device(f'cuda:{job_id % torch.cuda.device_count()}')

    params_fn = outdir / f"params_{fixed_path.stem}.npy"
    if args.skip_existing and params_fn.exists():
        print(f"Skipping {fixed_path}")
        return

    print(f"Processing {fixed_path}")
    moving_path = moving_dir / fixed_path.name
    if not moving_path.exists():
        print(f"Moving file not found: {moving_path}")
        return
    original_fixed = np.array(Image.open(fixed_path))
    original_moving = np.array(Image.open(moving_path))
    padded_moving = pad_moving(original_fixed, original_moving)

    def register(reg, n_iter, sample_rate, lr):
        reg.n_iter = n_iter
        fixed = downsample(original_fixed, sample_rate)[np.newaxis, np.newaxis]
        moving = downsample(np.array(padded_moving), sample_rate)[np.newaxis, np.newaxis]
        fixed = torch.tensor(fixed).float().to(device)
        moving = torch.tensor(moving).float().to(device)
        reg.to(device)
        reg.fit(moving, fixed, n_iter, lr)

    regs = []
    reg = RigidRegistration()
    reg.align_by_phase_correlation(original_moving, original_fixed)
    regs.append(('Phase correlation 1', reg, (10, 200, 0.01)))

    reg = RigidRegistration()
    reg.align_by_phase_correlation(original_moving, original_fixed)
    regs.append(('Phase correlation 3', reg, (10, 200, 0.03)))


    reg = RigidRegistration()
    reg.align_by_center_of_mass(original_moving, original_fixed)
    regs.append(('Center of mass', reg, (10, 200, 0.02)))

    for name, reg, (sample_rate, n_iter, lr) in regs:
        try:
            register(reg, n_iter, sample_rate, lr)
        except RuntimeError as e:
            print(f"Failed {name}: {e}")
            continue

    regs.sort(key=lambda x: x[1].losses[-1])
    reg = regs[0][1]
    print('Use', regs[0][0], reg.losses[-1])

    for (sample_rate, n_iter, lr) in [(5, 100, 0.002), (2, 100, 0.0002)]:
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
    np.save(params_fn, params)

    # break
from joblib import Parallel, delayed
Parallel(n_jobs=args.jobs)(delayed(run)(path, job_id)
                        for job_id, path in enumerate(tqdm(fixed_paths)))