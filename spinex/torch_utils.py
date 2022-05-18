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


class SoftArgmax(torch.nn.Module):
    '''
    Soft argmax for ND tensor. Input is a ND (batch, channel, i, j, ...) tensor.
    '''

    def __self__(self):
        super().__init__()

    def softmax(self, x):
        shape = x.shape
        flattened = x.view((shape[0], shape[1], -1))
        sm = F.softmax(flattened, dim=2)
        return sm.view(shape)

    def forward(self, x):
        ret = []
        sm = self.softmax(x)
        for axis in range(2, x.dim()):
            indices = torch.arange(x.shape[axis],
                                   device=x.device,
                                   dtype=torch.float32)
            if x.dim() > 3:  # 2d or more
                # reduce other axes
                sum_axis = tuple((i for i in range(2, x.dim()) if i != axis))
                summed = torch.sum(sm, axis=sum_axis)
            else:  # 1d tensor
                summed = sm
            summed = summed.view((x.shape[0] * x.shape[1], -1))
            center = torch.matmul(summed, indices).view(
                (x.shape[0], x.shape[1]))
            ret.append(torch.unsqueeze(center, -1))
        return torch.cat(ret, axis=-1)