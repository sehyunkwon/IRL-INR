import numpy as np
import torch
from torchvision import transforms

# counter-clockwise rotation
def make_coord(batch_size, resolution, theta, pred_center):
    # x coordinate array
    xgrid = np.linspace(-1, 1, resolution)
    ygrid = np.linspace(1, -1, resolution)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()
    x_coord = x_coord.expand(batch_size, resolution**2, 2) # x_coord.shape = b x (WxH) x 2, '2' is coordinate dimension
    x_coord = x_coord.to(theta.device)
    # print('x_coord',x_coord.shape)
    # print('pred_center', pred_center.shape)
    pred_center = pred_center.unsqueeze(1)

    x_coord = x_coord - pred_center
    # calculate rotation matrix counter clockwise
    rot = theta.data.new(batch_size,2,2).zero_()
    rot[:,0,0] = torch.cos(-theta)
    rot[:,0,1] = torch.sin(-theta)
    rot[:,1,0] = -torch.sin(-theta)
    rot[:,1,1] = torch.cos(-theta)
    # rotate coordinates by theta
    x_coord = torch.bmm(x_coord, rot)
    return x_coord

