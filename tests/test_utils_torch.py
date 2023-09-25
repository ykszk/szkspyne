import unittest
import numpy as np


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))


class TestStringMethods(unittest.TestCase):
    def test_accuracy(self):
        # from scipy.interpolate import RegularGridInterpolator
        import torch
        import szkspyne.torch_utils as torch_utils

        TOLERANCE = 0.2  # maybe later: TOO HIGH!?
        SIZE = 11
        MIN, MAX = -7, 7
        xs, ys = np.linspace(MIN, MAX, SIZE), np.linspace(MIN, MAX, SIZE)
        xy_grid = np.meshgrid(xs, ys)
        z = f(xy_grid[0], xy_grid[1])
        # rgi = RegularGridInterpolator((xs, ys), z, method='cubic')
        N_POINTS = 100
        yx = np.random.rand(2, N_POINTS) * (SIZE - 3) + 1
        # yx = np.array(([1,SIZE-2],[SIZE-3,5]))
        real_yx = yx / (SIZE - 1) * (MAX - MIN) + MIN
        true_values = f(real_yx[1], real_yx[0])

        # scipy_values = rgi(real_yx.T)

        t_z = torch.FloatTensor(z)
        t_yx = torch.FloatTensor(yx).T
        t_interpolated = torch_utils.bicubic(t_z, t_yx)

        diff = t_interpolated.numpy() - true_values
        self.assertFalse(np.any(np.abs(diff) > TOLERANCE))


# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# import szkspyne.torch_utils as torch_utils

# SIZE=11
# MIN, MAX = -7, 7
# FINE_SIZE = 200
# fine_xs, fine_ys = np.linspace(0, SIZE-1, FINE_SIZE), np.linspace(0, SIZE-1, FINE_SIZE)
# # fine_xs, fine_ys = np.linspace(MIN, MAX, FINE_SIZE), np.linspace(MIN, MAX, FINE_SIZE)
# fine_grid = np.meshgrid(np.linspace(MIN, MAX, FINE_SIZE), np.linspace(MIN, MAX, FINE_SIZE))
# fine_z = f(fine_grid[0], fine_grid[1])
# plt.imshow(fine_z)
# plt.colorbar()
# plt.show()

# #%%
# fine_grid = np.meshgrid(fine_xs, fine_ys)
# fine_yx = np.stack((fine_grid[1], fine_grid[0]), axis=-1).reshape(-1, 2)
# fine_yx.shape
# t_fine_yx = torch.FloatTensor(fine_yx)

# xs, ys = np.linspace(MIN, MAX, SIZE), np.linspace(MIN, MAX, SIZE)
# xy_grid = np.meshgrid(xs, ys)
# z = f(xy_grid[0], xy_grid[1])
# # rgi = RegularGridInterpolator((xs, ys), z, method='cubic')
# N_POINTS = 100
# yx = np.random.rand(2, N_POINTS) * (SIZE-3) + 1
# # yx = np.array(([1,SIZE-2],[SIZE-3,5]))
# real_yx = yx / (SIZE-1) * (MAX-MIN)+MIN
# true_values = f(real_yx[1], real_yx[0])

# # scipy_values = rgi(real_yx.T)

# t_z = torch.FloatTensor(z)
# t_yx = torch.FloatTensor(yx).T
# t_interpolated = torch_utils.bicubic(t_z, t_yx)

# t_fine_interpolated = torch_utils.bicubic(t_z, t_fine_yx).numpy()
# t_fine_img = t_fine_interpolated.reshape((FINE_SIZE, FINE_SIZE))
# plt.imshow(t_fine_img)
# plt.colorbar()
# plt.show()

# #%%
# from scipy.interpolate import RegularGridInterpolator
# rgi = RegularGridInterpolator((np.arange(SIZE), np.arange(SIZE)), z, method='cubic')
# scipy_fine_interpolated = rgi(fine_yx)
# scipy_fine_img = scipy_fine_interpolated.reshape((FINE_SIZE, FINE_SIZE))
# plt.imshow(scipy_fine_img)
# plt.colorbar()
# plt.show()

# # %%
# import seaborn as sns
# diff = fine_z - t_fine_img
# plt.imshow(diff)
# plt.title('true - mine')
# plt.colorbar()
# plt.show()
# sns.histplot(diff.ravel())
# rms = np.sqrt((diff.ravel()**2).sum())
# plt.title(f'RMS={rms}')
# plt.show()

# diff = fine_z - scipy_fine_img
# plt.imshow(diff)
# plt.title('true - scipy')
# plt.colorbar()
# plt.show()
# sns.histplot(diff.ravel())
# rms = np.sqrt((diff.ravel()**2).sum())
# plt.title(f'RMS={rms}')
# plt.show()

# #%%
# y = len(t_fine_img)//2
# plt.plot(fine_z[y, :], label='true')
# plt.plot(t_fine_img[y, :], label='mine')
# plt.plot(scipy_fine_img[y, :], label='scipy')
# plt.scatter(np.linspace(0, FINE_SIZE, SIZE), z[len(z)//2, :])
# plt.legend()
# plt.show()
