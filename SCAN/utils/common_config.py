"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from data.augment import Augment, Cutout
from utils.collate import collate_custom
from data.custom_dataset import CustomImageDataset

 
def get_criterion(p):
    if p['criterion'] == 'simclr':
        from losses.losses import SimCLRLoss
        criterion = SimCLRLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'scan':
        from losses.losses import SCANLoss
        criterion = SCANLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'confidence-cross-entropy':
        from losses.losses import ConfidenceBasedCE
        criterion = ConfidenceBasedCE(p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512

    elif p['backbone'] == 'resnet50':
        return 2048

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'resnet18':
        from models.resnet18 import resnet18
        backbone = resnet18()
        print('Backbone: ResNet18')
        
    elif p['backbone'] == 'resnet18+mlp':
        from models.resnet18_mlp import resnet18
        backbone = resnet18(p['model_kwargs']['num_layers'], p['model_kwargs']['output_dim'])
        # for name, param in backbone['backbone'].named_parameters():
	    #     print(name)
        if p['pretrained'] == True:
        # Load pretrained backbone(=encoder of INR)
            path = p['model_kwargs']['pretrained_path']
            checkpoint = torch.load(f'./{path}', map_location='cpu')
            # for name in checkpoint.keys():
	        #     print(name)
            print('pretrained_path is: ', path)
            
            # if p['model_kwargs']['num_layers'] == 1:
            #     checkpoint2 = {'fc_layer.0.weight':checkpoint['fc_layer.0.weight'][2:], 'fc_layer.0.bias':checkpoint['fc_layer.0.bias'][2:]}
            #     del checkpoint['fc_layer.0.weight']
            #     del checkpoint['fc_layer.0.bias']
            # elif p['model_kwargs']['num_layers'] == 2:
            #     checkpoint2 = {'fc_layer.2.weight':checkpoint['fc_layer.2.weight'][2:], 'fc_layer.2.bias':checkpoint['fc_layer.2.bias'][2:]}
            #     del checkpoint['fc_layer.2.weight']
            #     del checkpoint['fc_layer.2.bias']
            # elif p['model_kwargs']['num_layers'] == 3:
            #     checkpoint2 = {'fc_layer.4.weight':checkpoint['fc_layer.4.weight'][2:], 'fc_layer.4.bias':checkpoint['fc_layer.4.bias'][2:]}
            #     del checkpoint['fc_layer.4.weight']
            #     del checkpoint['fc_layer.4.bias']

            # n = 2*(p['model_kwargs']['num_layers']-1)
            # checkpoint2 = {f'fc_layer.{n}.weight':checkpoint[f'fc_layer.{n}.weight'][2:], f'{n}.bias':checkpoint[f'fc_layer.{n}.bias'][2:]}
            # del checkpoint[f'fc_layer.{n}.weight']
            # del checkpoint[f'fc_layer.{n}.bias']
            
            n = p['model_kwargs']['num_layers']+1
            checkpoint2 = {f'fc_layer.{n}.weight':checkpoint[f'fc_layer.{n}.weight'][2:], f'{n}.bias':checkpoint[f'fc_layer.{n}.bias'][2:]}
            del checkpoint[f'fc_layer.{n}.weight']
            del checkpoint[f'fc_layer.{n}.bias']
            # else:
            #     ValueError('Invalid num_layers'.format(p['model_kwargs']['num_layers']))
            backbone['backbone'].load_state_dict(checkpoint, strict=False)
            backbone['backbone'].load_state_dict(checkpoint2, strict=False)
            print('Load pretrained backbone!')
        else:
            print('Training from scratch')

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    if p['setup'] == 'mining':
        # model = backbone['backbone']
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, p['model_kwargs']['head'], p['model_kwargs']['features_dim'])

    elif p['setup'] in ['simclr', 'moco']:
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, p['model_kwargs']['head'], p['model_kwargs']['features_dim'])

    elif p['setup'] in ['scan', 'selflabel']:
        from models.models import ClusteringModel
        if p['setup'] == 'selflabel':
            assert(p['num_heads'] == 1)
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])

    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        print(pretrain_path)
        state = torch.load(pretrain_path, map_location='cpu')
        
        if p['setup'] == 'scan': # Weights are supposed to be transfered from contrastive training
            # print(model)
            missing = model.load_state_dict(state, strict=False)
            print(set(missing[1]))
            assert(set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias', 
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                or set(missing[1]) == {
                'contrastive_head.weight', 'contrastive_head.bias'})

        elif p['setup'] == 'selflabel': # Weights are supposed to be transfered from scan 
            # We only continue with the best head (pop all heads first, then copy back the best head)
            model_state = state['model']
            all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
            best_head_weight = model_state['cluster_head.%d.weight' %(state['head'])]
            best_head_bias = model_state['cluster_head.%d.bias' %(state['head'])]
            for k in all_heads:
                model_state.pop(k)

            model_state['cluster_head.0.weight'] = best_head_weight
            model_state['cluster_head.0.bias'] = best_head_bias
 
            missing = model.load_state_dict(model_state, strict=True)

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model

def get_train_dataset(p, transform=None, to_augmented_dataset=False,
                        to_neighbors_dataset=False, split=None):
    if p['train_db_name'] == 'mnist':
        ckp = torch.load('../rotated_mnist.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)

    elif p['train_db_name'] == 'mnist-u':
        ckp = torch.load('../MNIST-U.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)
        
    
    elif p['train_db_name'] == 'fashionmnist':
        ckp = torch.load('rotated_fashionmnist.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']
        
        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)
    
    elif p['train_db_name'] == 'wm811k':
        ckp = torch.load('rotated_wm811k.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)

    elif p['train_db_name'] == 'hynix':
        ckp = torch.load('hynix.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)
    
    elif p['train_db_name'] == 'plankton':
        ckp = torch.load('rotated_plankton.pt')
        train_image_set = ckp['train_image']
        train_label_set = ckp['train_label']

        dataset = CustomImageDataset(train_image_set, train_label_set, transform=transform)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        print('augmentation!')
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        print('to_neighbors_dataset!')
        indices = np.load(p['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])
    
    return dataset


def get_val_dataset(p, transform=None, to_neighbors_dataset=False):
    # Base dataset    
    if p['train_db_name'] == 'mnist':
        ckp = torch.load('rotated_mnist.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)

    elif p['train_db_name'] == 'mnist-u':
        ckp = torch.load('../MNIST-U.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)
    
    elif p['train_db_name'] == 'fashionmnist':
        ckp = torch.load('rotated_fashionmnist.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)

    elif p['train_db_name'] == 'wm811k':
        ckp = torch.load('rotated_wm811k.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)

    elif p['train_db_name'] == 'hynix':
        ckp = torch.load('hynix.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)
    
    elif p['train_db_name'] == 'plankton':
        ckp = torch.load('rotated_plankton.pt')
        test_image_set = ckp['test_image']
        test_label_set = ckp['test_label']

        dataset = CustomImageDataset(test_image_set, test_label_set, transform=transform)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes) 
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        print('to_neighbors_dataset!')
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors']) # Only use 5

    return dataset

def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'None':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        return transforms.Compose([])

    elif p['augmentation_strategy'] == 'standard':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    
    elif p['augmentation_strategy'] == 'simclr':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Augmentatio아n strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p'])
            ])

    elif p['augmentation_strategy'] == 'rotation+translation':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Augmentatio아n strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.07, 0.07,), scale=None, shear=None, fill=0, center=None),
            transforms.RandomRotation(180)
            ])
    
    elif p['augmentation_strategy'] == 'rotation':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Augmentatio아n strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomRotation(180)
            ])

    elif p['augmentation_strategy'] == 'simclr+rotation+translation':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Augmentatio아n strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.07, 0.07,), scale=None, shear=None, fill=0, center=None),
            transforms.RandomRotation(180),
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p'])
            ])

    elif p['augmentation_strategy'] == 'simclr+rotation':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Augmentatio아n strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p'])
            ])
    
    elif p['augmentation_strategy'] == 'ours':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Augmentation strategy from our paper 
        return transforms.Compose([
            transforms.Resize(p['resolution']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])])

    elif p['augmentation_strategy'] == 'ours+rotation':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Augmentation strategy from our paper 
        return transforms.Compose([
            transforms.RandomRotation(180),
            transforms.Resize(p['resolution']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])
                ])

    elif p['augmentation_strategy'] == 'ours+rotation+translation':
        print('augmentation_strategy: ', p['augmentation_strategy'])
        # Augmentation strategy from our paper 
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.07, 0.07,), scale=None, shear=None, fill=0, center=None),
            transforms.RandomRotation(180),
            transforms.Resize(p['resolution']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])
                ])
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    return transforms.Compose([])


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()
                

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
