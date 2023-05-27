import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms

import cv2
from tqdm import tqdm
import time
import json
import os
import sys

from resnet_gray import resnet18
from RFF import FourierFeatures
from make_coord import make_coord
from models import FunctionRepresentation, HyperNetwork
from utils import find_center, get_theta
from data.dataloader import get_train_dataset, get_val_dataset, get_train_dataloader, get_val_dataloader

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main.py <config_path>"))
config_path = sys.argv[1]

# Open config file
with open(config_path) as f:
    config = json.load(f)

device= config["device"]

# Create a folder to store experiment results
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
directory = "{}_{}_experiment".format(timestamp, config["dataset"])
if not os.path.exists(directory):
    os.makedirs(directory)
# Save config file in experiment directory
with open(directory + '/config.json', 'w') as f:
    json.dump(config, f)

# Dataset
train_dataset = get_train_dataset(config['dataset'], transform=None) # Split is for stl-10
val_dataset = get_val_dataset(config['dataset'])
train_dataloader = get_train_dataloader(config, train_dataset)
val_dataloader = get_val_dataloader(config, val_dataset)
print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

# Encoder
encoder = resnet18(config['feature_dim'], config["num_layers"], config["latent_dim"], config["resid_mlp"])
encoder= encoder.to(device)

# Random Fourier Feature
frequency_matrix = torch.normal(mean=torch.zeros(config["num_frequencies"], 2), std=2.0).to(device)
fourier_feature = FourierFeatures(frequency_matrix)

# FunctionRepresentation
function_representation = FunctionRepresentation(config["feature_dim"], config["layer_sizes"], config["num_frequencies"]).to(device)

# HyperNetwork
hypernetwork = HyperNetwork(function_representation, config["latent_dim"], config["hypernet_layer_sizes"]).to(device)

# Calculate Model Size
total_params = sum(p.numel() for p in encoder.parameters())
print('total parameters of Encoder: %.2f M' % (total_params / 1000 / 1000))

total_params = sum(p.numel() for p in function_representation.parameters())
print('total parameters of Function Representation: %.2f M' % (total_params / 1000 / 1000))

total_params = sum(p.numel() for p in hypernetwork.parameters())
print('total parameters of HyperNetwork: %.2f M' % (total_params / 1000 / 1000))

# Optimizer
optimizer_hyper = torch.optim.Adam(hypernetwork.forward_layers.parameters(),
            lr=config["lr_hypernetwork"], betas=(0.5, 0.999))

optimizer_encoder = torch.optim.Adam(encoder.parameters(),
            lr=config["lr_encoder"], betas=(0.5, 0.999))
           

# Loss function
mse = nn.MSELoss()
cos = nn.CosineSimilarity(dim=1)

for epoch in tqdm(range(config["epochs"])):
    train_loss = 0.
    train_pred = 0.
    train_recon = 0.
    train_consis = 0.
    for i, dict in tqdm(enumerate(train_dataloader)):
        time.sleep(0.1)
        images = dict['image']
        images = transforms.RandomRotation(180)(images)
        b = images.shape[0]
        images = images.to(device)


        ### SYMMETRY BREAKING LOSS###

        # Encoder takes image as an input
        z = encoder(images)
        # Encoder outputs ((theta, tau), z). However, (theta and tau) can be replaced by predicted center of mass by symmetry breaking
        pred_center, z1 = z[:, 0:2], z[:, 2:]
        # Calculate true center of mass of given image
        true_center = find_center(images)
        # Calculate symmetry breaking loss
        loss_pred = mse(pred_center, true_center)
        # Obtaining predicted theta(=theta hat). Rotation degree theta can be recovered from center of mass
        pred_theta = get_theta(pred_center) 


        ### RECONSTRUCTION LOSS ###

        # Input coordinates of INR are transformed by predicted theta and tau
        if config["dataset"] in ["wm811k", "plankton", "5hdb"]:
            x_coord = make_coord(b, config["resolution"], -pred_theta, 0*pred_center)  # We assume that only rotations have been applied to these datasets.
        elif config["dataset"] in ["MNIST-U", "galaxy_zoo", "dsprites"]:
            x_coord = make_coord(b, config["resolution"], -pred_theta, pred_center)  # We assume that rotations & translations have been applied to these datasets.
        # Coordinates are transformed to fourier feature
        fourier_coords = fourier_feature(x_coord).to(device)
        # Generate parameters of INR
        all_weights, all_biases = hypernetwork(z1) 

        generated_images = torch.empty(b, config["resolution"]**2, config["feature_dim"]).to(device)
        for j in range(b):
            generated_images[j] = function_representation(fourier_coords[j], all_weights[j], all_biases[j]).view(config["resolution"]**2, config["feature_dim"])

        images = images.view(b, -1)
        # Calculate reconstruction loss
        loss_recon = mse(images, generated_images.view(b, -1))


        ### CONSISTENCY LOSS ###

        # Make augmented image. These images are utlized for making another latent vector z2
        rotated_images = transforms.RandomRotation(180)(images.view(-1,config["feature_dim"],config["resolution"],config["resolution"]))

        if config["dataset"] in ["wm811k", "plankton", "5hdb"]:
            z2 = encoder(rotated_images)[:, 2:]
        elif config["dataset"] in ["MNIST-U", "galaxy_zoo", "dsprites"]:
            # We assume that rotations & translations have been applied to these datasets. So we apply random translation for augmentation
            translated_images = torch.empty(rotated_images.shape)
            for k in range(b):
                d1, d2 = 5*torch.randn(1).item(), 5*torch.randn(1).item()
                M = np.float32([[1,0,d1],[0,1,d2]])
                rotated_image = rotated_images[k].cpu().numpy()
                rotated_image = rotated_image.reshape(config["resolution"],config["resolution"],config["feature_dim"])
                translated_image = cv2.warpAffine(rotated_image, M, (0, 0))
                translated_image = transforms.ToTensor()(translated_image)
                translated_images[k] = translated_image
            translated_images = translated_images.to(device)
            z2 = encoder(translated_images)[:, 2:]

        # calculate cosine similarity
        if config["consistency_loss"] == "cos":
            loss_consis = 1 - cos(z1,z2)
        else:
            loss_consis = mse(z1, z2)

        # Final objective function.
        loss = 15*loss_pred + loss_recon + config["coefficient"]*loss_consis.mean()

        loss.backward()

        optimizer_hyper.step()
        optimizer_encoder.step()

        optimizer_hyper.zero_grad()
        optimizer_encoder.zero_grad()

        train_loss += loss.item() * b
        train_pred += loss_pred.item() * b
        train_recon += loss_recon.item() * b
        train_consis += (loss_consis.mean()).item() * b

    print(f"{epoch+1}th Epoch, recon_loss:{train_recon/len(train_dataset)}, pred_loss:{train_pred/len(train_dataset)}, consis_loss:{train_consis/len(train_dataset)}, total_loss:{train_loss/len(train_dataset)},")
    
    # Save model
    torch.save(encoder.state_dict(), directory+f'/encoder_{epoch+1}.pt')
    torch.save(hypernetwork.state_dict(), directory+f'/hypernet_{epoch+1}.pt')
    torch.save(fourier_feature.state_dict(), directory+f'/rff_{epoch+1}.pt')
    
    # Save reconstructed image.
    if epoch % 1 == 0:
        for dict in val_dataloader:
            images = dict['image']
            b = images.shape[0]
            images = images.to(device)
            z = encoder(images)
            pred_center, z = z[:,0:2], z[:,2:]
            true_center = find_center(images)
            theta = get_theta(pred_center)
            if config["dataset"] in ["wm811k", "plankton", "5hdb"]:
                x_coord = make_coord(b, config["resolution"], -theta, 0*pred_center)
            elif config["dataset"] in ["MNIST-U", "galaxy_zoo", "dsprites"]:
                x_coord = make_coord(b, config["resolution"], -theta, pred_center)
            # make random fourier feature
            fourier_coords = fourier_feature(x_coord).to(device)
            all_weights, all_biases = hypernetwork(z) 

            generated_images = torch.empty(b, config["resolution"]**2, config["feature_dim"]).to(device)
            for j in range(b):
                # Set weights and biases as predicted by hypernetwork & input to fourier coordinates and then generate images
                generated_images[j] = function_representation(fourier_coords[j], all_weights[j], all_biases[j]).view(config["resolution"]**2, config["feature_dim"])

            # Images for Appendix E
            save_image(images.view(b, config["feature_dim"], config["resolution"], config["resolution"]), directory+'/valset_{}th_input.png'.format(epoch+1))
            save_image(generated_images.view(b, config["feature_dim"], config["resolution"], config["resolution"]), directory+'/valset_{}th_output.png'.format(epoch+1))

            # rotate just 'one' single image
            rotated_images = torch.empty(16*8, config["feature_dim"], config["resolution"], config["resolution"]).to(device)
            a = 0
            for j in range(16):
                for i in range(8):
                    rotated_images[8*j+i] = transforms.functional.rotate(images[a], i*(360/8))
                a += 1
            save_image(rotated_images, directory+'/valset_{}th_rotated_input.png'.format(epoch+1))

            z = encoder(rotated_images)
            pred_center, z = z[:,0:2], z[:,2:]
            theta = get_theta(pred_center)

            if config["dataset"] in ["wm811k", "plankton", "5hdb"]:
                x_coord = make_coord(b, config["resolution"], -theta*0, 0*pred_center)
            elif config["dataset"] in ["MNIST-U", "galaxy_zoo", "dsprites"]:
                x_coord = make_coord(b, config["resolution"], -theta*0, 0*pred_center)
            
            fourier_coords = fourier_feature(x_coord).to(device)
            all_weights, all_biases = hypernetwork(z) 

            generated_images = torch.empty(16*8, config["resolution"]**2, config["feature_dim"]).to(device)
            for j in range(16*8):
                generated_images[j] = function_representation(fourier_coords[j], all_weights[j], all_biases[j]).view(config["resolution"]**2, config["feature_dim"])

            # Images for Appendix F
            save_image(generated_images.view(16*8, config["feature_dim"], config["resolution"], config["resolution"]), directory+'/valset_{}th_rotated_output.png'.format(epoch+1))
            break