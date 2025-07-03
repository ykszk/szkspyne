import torch
import torch.nn.functional as F


def line_dist_gauss(shape, p1, p2, sigma, device, dtype):
    '''
    Distance from line segment + Gaussian function.
    Note: max of return value is 1.

    distance: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment

    Args:
        shape : output shape
        p1 : first point
        p2 : second point
        sigma : sigma
    '''
    p2p1 = p2 - p1
    l = (p2p1**2).sum()
    ys = torch.arange(0, shape[0], device=device, dtype=dtype)
    xs = torch.arange(0, shape[1], device=device, dtype=dtype)
    ys = ys.view(shape[0], 1)
    xs = xs.view(1, shape[1])
    dot_y = ys - p1[0]
    dot_y *= p2p1[0] / l
    dot_x = xs - p1[1]
    dot = torch.add(dot_y, dot_x, alpha=p2p1[1] / l)  # broadcast
    t = torch.clamp(dot, 0, 1, out=dot)
    ys -= p1[0]
    d_y = torch.sub(ys, t, alpha=p2p1[0])  # broadcast
    torch.pow(d_y, 2, out=d_y)
    xs -= p1[1]
    d_x = torch.sub(xs, t, alpha=p2p1[1], out=t)  # broadcast # use t as out
    torch.pow(d_x, 2, out=d_x)
    dist = torch.add(d_y, d_x, out=d_y)
    return torch.exp((-0.5 * torch.reciprocal(sigma**2)) * dist)


def gaussian2d(shape, mu, sigma, device, dtype):
    '''
    Generate PDF(heatmap) for 2D gaussian.
    Note: max of return value is 1.

    Args:
        shape (array) : output shape
        mu (array) : mean/center
        sigma (array or scalar): standard deviation
    '''
    vs = []
    for i in range(2):
        v = torch.arange(0, shape[i], device=device, dtype=dtype)
        v -= mu[i]
        torch.pow(v, 2, out=v)
        v *= -0.5 * torch.reciprocal(sigma**2)
        v = torch.exp(v)
        vs.append(v)
    gauss = vs[0].view((vs[0].shape[0], 1)) * vs[1].view((1, vs[1].shape[0]))
    return gauss


def lerp2d(arr, idx):
    '''
    Args:
        arr: 2d torch tensor
        idx: [[y, x], ...]
    '''
    idx = torch.clip(idx, torch.tensor(0, device=idx.device), torch.tensor(arr.shape) - 2)
    ia = torch.floor(idx).long()
    ib = ia + 1
    a, b = arr[ia[:, 0], ia[:, 1]], arr[ia[:, 0], ib[:, 1]]
    c, d = arr[ib[:, 0], ia[:, 1]], arr[ib[:, 0], ib[:, 1]]
    wx = idx[:, 1] - ia[:, 1]
    wy = idx[:, 0] - ia[:, 0]
    ab, cd = torch.lerp(a, b, wx), torch.lerp(c, d, wx)
    return torch.lerp(ab, cd, wy)


Cx = (
    torch.FloatTensor([[-1.0, 3.0, -2.0, 0.0], [3.0, -6.0, -3.0, 6.0], [-3.0, 3.0, 6.0, 0.0], [1.0, 0.0, -1.0, 0.0]])
    / 6
)
Cy = Cx.T
Cx, Cy = Cx.unsqueeze(0), Cy.unsqueeze(0)  # for batch operations


def bicubic(arr, yxs):
    '''
    Bicubic interpolation

    - [Bicubic interpolation - Wikipedia](https://en.wikipedia.org/wiki/Bicubic_interpolation)
    - [numpy - Bicubic interpolation Python - Stack Overflow](https://stackoverflow.com/questions/52700878/bicubic-interpolation-python#answer-52702678)
    Args:
        arr: 2d torch tensor
        yxs (tensor): [[y, x], ...]
    '''
    yx00 = torch.floor(yxs).long()
    ty = yxs[:, 0] - yx00[:, 0]
    tx = yxs[:, 1] - yx00[:, 1]
    zs = []
    clip_min = torch.tensor(0, device=yxs.device)
    clip_y_max = torch.tensor(arr.shape[0] - 1, device=yxs.device)
    clip_x_max = torch.tensor(arr.shape[1] - 1, device=yxs.device)
    for dy in [-1, 0, 1, 2]:
        for dx in [-1, 0, 1, 2]:
            ys = torch.clip(yx00[:, 0] + dy, clip_min, clip_y_max)
            xs = torch.clip(yx00[:, 1] + dx, clip_min, clip_x_max)
            zs.append(arr[ys, xs])

    device = yxs.device
    zs = torch.stack(zs, dim=1).reshape((yxs.shape[0], 4, 4)).to(device=device)

    px = torch.stack([tx**3, tx**2, tx, torch.ones_like(tx)], dim=1).unsqueeze(2).to(device=device)
    py = torch.stack([ty**3, ty**2, ty, torch.ones_like(ty)], dim=1).unsqueeze(1).to(device=device)

    p = py @ Cy.to(device=device) @ zs @ Cx.to(device=device) @ px
    return p.squeeze(1).squeeze(1)
