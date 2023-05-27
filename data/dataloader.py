import os
import math
import numpy as np
import torch
from data.collate import collate_custom
from data.custom_dataset import CustomImageDataset

def get_train_dataset(dataset, transform=None, to_neighbors_dataset=False):
    if dataset == 'galaxy_zoo':
        ckp = torch.load(f'{dataset}.pt')
        train_image_set = ckp['train_image']

        dataset = CustomImageDataset(train_image_set, transform=transform)
    
    elif dataset == 'MNIST-U':
        ckp = torch.load(f'{dataset}.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)

    elif dataset == 'removed_MNIST-N':
        ckp = torch.load(f'{dataset}.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)

    elif dataset == 'removed_MNIST-U':
        ckp = torch.load(f'{dataset}.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)
        
    
    elif dataset == 'fashionmnist':
        ckp = torch.load('rotated_fashionmnist.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']
        
        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)
    
    elif dataset == 'wm811k':
        ckp = torch.load('wm811k.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)
    
    elif dataset == 'plankton':
        ckp = torch.load('plankton.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)

    elif dataset == 'dsprites':
        ckp = torch.load('dsprites.pt')
        train_image_set = ckp['train_image']

        dataset = CustomImageDataset(train_image_set, label=None, transform=transform)

    elif dataset == '5hdb':
        ckp = torch.load('5hdb.pt')
        train_image_set = ckp['train_image']

        dataset = CustomImageDataset(train_image_set, label=None, transform=transform)
   
    else:
        raise ValueError('Invalid train dataset {}'.format(dataset))
    
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        print('to_neighbors_dataset!')
        indices = np.load(p['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])
    
    return dataset


def get_val_dataset(dataset, transform=None, to_neighbors_dataset=False):
    # Base dataset    
    if dataset == 'galaxy_zoo':
        ckp = torch.load(f'{dataset}.pt')
        test_image_set = ckp['test_image']

        dataset = CustomImageDataset(test_image_set, transform=transform)
    
    elif dataset == 'MNIST-U':
        ckp = torch.load(f'{dataset}.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)

    elif dataset == 'wm811k':
        ckp = torch.load('wm811k.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)
    
    elif dataset == 'plankton':
        ckp = torch.load('plankton.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)
    
    elif dataset == 'dsprites':
        ckp = torch.load('dsprites.pt')
        test_image_set = ckp['test_image']

        dataset = CustomImageDataset(test_image_set, label=None, transform=transform)

    elif dataset == '5hdb':
        ckp = torch.load('5hdb.pt')
        test_image_set = ckp['test_image']

        dataset = CustomImageDataset(test_image_set, label=None, transform=transform)

    else:
        raise ValueError('Invalid train dataset {}'.format(dataset))
    
    # Wrap into other dataset (__getitem__ changes) 
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        print('to_neighbors_dataset!')
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, 5) # Only use 5

    return dataset


def get_train_dataloader(p, dataset, shuffle=True):
    return torch.utils.data.DataLoader(dataset, num_workers=8, 
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=shuffle)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=0,
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)