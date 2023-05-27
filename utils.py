import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import numpy as np
from scipy.optimize import linear_sum_assignment

def cluster_acc(y_true, y_pred):
    """
    Arguments
        y: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
    Return
        accuracy
    """
    y_true = y_true.astype(np.int64)
    # print('y_true',y_true.size())
    # print('y_pred', y_pred.size())
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    mapping = linear_sum_assignment(w.max() - w) 

    sum_ = 0
    for i in range(len(mapping[0])):
        sum_ += w[mapping[0][i]][mapping[1][i]]

    return mapping, (sum_/y_pred.shape[0])

def rotate_reference_degree(image, coordinate):
    m_x, m_y = coordinate[0], coordinate[1]
    if m_x > 0:
        cos = m_y / ((torch.sqrt(m_x**2+m_y**2))+1e-6)
        theta = torch.arccos(cos).item()
        image = transforms.functional.rotate(image, (180/3.14)*theta)
    else:
        cos = m_y / ((torch.sqrt(m_x**2+m_y**2))+1e-6)
        theta = torch.arccos(cos).item()
        image = transforms.functional.rotate(image, -(180/3.14)*theta)
    return image

def find_center(images):
    if images.shape[1] == 3:
        images = transforms.Grayscale(num_output_channels=1)(images)
    device = images.device
    b = images.shape[0]
    w = images.shape[2]
    center = torch.empty(b, 2).to(device)
    for j in range(b):
        image = images[j]
        m = torch.sum(image)
        center[j][0] = torch.sum(torch.matmul(image.view(w, w), torch.linspace(-1, 1, steps=w).view(w, 1).to(device))) / m
        center[j][1] = torch.sum(torch.matmul(torch.linspace(1, -1, steps=w).to(device), image.view(w, w))) / m
    return center


def get_theta(coordinates):
    # for loop is not efficient. fix later.
    b = coordinates.shape[0]
    m_x, m_y = coordinates[:,0], coordinates[:,1]

    theta = torch.empty(b).to(coordinates.device)
    for i in range(b):
        if m_x[i] > 0:
            cos = m_y[i] / ((torch.sqrt(m_x[i]**2+m_y[i]**2))+1e-6)
            theta[i] = torch.arccos(cos)
        else:
            cos = m_y[i] / ((torch.sqrt(m_x[i]**2+m_y[i]**2))+1e-6)
            theta[i] = -torch.arccos(cos)

    return theta

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img