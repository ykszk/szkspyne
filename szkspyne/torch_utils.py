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
    return torch.exp((-.5 * torch.reciprocal(sigma**2)) * dist)


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
        v *= -.5 * torch.reciprocal(sigma**2)
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
    ia = torch.floor(idx).long()
    ib = ia + 1
    a, b = arr[ia[:, 0], ia[:, 1]], arr[ia[:, 0], ib[:, 1]]
    c, d = arr[ib[:, 0], ia[:, 1]], arr[ib[:, 0], ib[:, 1]]
    wx = idx[:, 1] - ia[:, 1]
    wy = idx[:, 0] - ia[:, 0]
    ab, cd = torch.lerp(a, b, wx), torch.lerp(c, d, wx)
    return torch.lerp(ab, cd, wy)


Cx = torch.FloatTensor([[-1., 3., -2., 0.], [3., -6., -3., 6.],
                        [-3., 3., 6., 0.], [1., 0., -1., 0.]]) / 6
Cy = Cx.T
Cx, Cy = Cx.unsqueeze(0), Cy.unsqueeze(0)  # for batch operations


def bicubic(arr, xys):
    '''
    [Bicubic interpolation - Wikipedia](https://en.wikipedia.org/wiki/Bicubic_interpolation)
    Args:
        arr: 2d torch tensor
        idx: [[y, x], ...]
    '''
    xy00 = torch.floor(xys).long()
    ty = xys[:, 0] - xy00[:, 0]
    tx = xys[:, 1] - xy00[:, 1]
    zs = []
    for dy in [-1, 0, 1, 2]:
        for dx in [-1, 0, 1, 2]:
            ys = torch.clip(xy00[:, 0] + dy, 0, arr.shape[0] - 1)
            xs = torch.clip(xy00[:, 1] + dx, 0, arr.shape[1] - 1)
            zs.append(arr[ys, xs])

    zs = torch.stack(zs, dim=1).reshape((xys.shape[0], 4, 4))

    px = torch.stack([tx**3, tx**2, tx, tx**0], dim=1).unsqueeze(2)
    py = torch.stack([ty**3, ty**2, ty, ty**0], dim=1).unsqueeze(1)

    p = py @ Cy @ zs @ Cx @ px
    return p.squeeze(1).squeeze(1)