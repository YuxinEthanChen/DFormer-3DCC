import random
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from refocus_augmentation import RefocusImageAugmentation
from augs_3d_cc import motionkornia_blur_3d, zoomkornia_blur_3d, flash_3d, fog_3d, fog_3d_val, defocus_blur_3d_random
from augs_2d_cc import jpeg_compression, pixelate, shot_noise, impulse_noise, defocus_blur, fog, zoom_blur

import numpy as np
import PIL
from PIL import Image, ImageOps
from torchvision import transforms

import pdb
try:
    from kornia.augmentation import *
except:
    print("Error importing kornia augmentation")

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

def iso_noise(x, severity=1):
    device = x.device

    c_poisson = 25  # ISO doesn't affect photon noise

    x = x / 255.
    x = x.to(device)  # Ensure x is on the correct device
    poisson_input = x * c_poisson
    poisson_output = torch.poisson(poisson_input)
    x = torch.clamp(poisson_output / c_poisson, 0, 1) * 255

    c_gauss = 0.7 * torch.tensor([.08, .12, 0.18, 0.26, 0.38], device=device)[severity - 1]  # ISO increases electronic noise
    x = x / 255.
    x = torch.clamp(x + torch.randn_like(x, device=device) * c_gauss, 0, 1) * 255

    return x

def poisson_gaussian_noise(x, severity=1):
    device = x.device

    c_poisson = 10 * torch.tensor([60, 25, 12, 5, 3], device=device)[severity - 1]

    x = x / 255.
    x = x.to(device)  # Ensure x is on the correct device
    poisson_input = x * c_poisson
    poisson_output = torch.poisson(poisson_input)
    x = torch.clamp(poisson_output / c_poisson, 0, 1) * 255

    c_gauss = 0.1 * torch.tensor([.08, .12, 0.18, 0.26, 0.38], device=device)[severity - 1]
    x = x / 255.
    x = torch.clamp(x + torch.randn_like(x, device=device) * c_gauss, 0, 1) * 255

    return x

def imadjust(x, a, b, c, d, gamma=1):
    device = x.device

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y.to(device)

def low_light(x, severity=1):
    device = x.device

    c = torch.tensor([.60, .50, 0.40, 0.30, 0.20, 0.1, 0.05], device=device)[severity - 1]
    x = x / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2.).to(device) * 255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=3)
    return x_scaled

# def color_quant(x, severity=1):
#     device = x.device

#     # Ensure severity is within an acceptable range
#     severity = min(max(severity, 1), 5)
#     bits = 6 - severity
#     batch_size = x.shape[0]
#     quantized_images = []

#     for i in range(batch_size):
#         img = x[i].cpu().clamp(0, 1)  # Ensure the tensor is in [0, 1] range
#         img_pil = transforms.ToPILImage()(img)
#         img_pil = ImageOps.posterize(img_pil, bits)
#         img_tensor = transforms.ToTensor()(img_pil).to(device)  # Convert back to tensor and move to original device
        
#         # Check for potential all-black images and handle
#         if img_tensor.max() == 0:
#             print(f"Image {i} is completely black after quantization; adjusting severity.")
#             img_tensor = x[i].clamp(0, 1)  # Revert to the original image if posterization fails
        
#         quantized_images.append(img_tensor)

#     x = torch.stack(quantized_images).to(device)
#     return x

def random_block_mask(images, min_m=2, max_m=10, min_n=2, max_n=10):
    """
    Randomly separates the image into m by n boxes, then randomly blocks
    a random number of boxes by turning them black. `m` and `n` are randomly generated.

    Parameters:
    images (torch.Tensor): Batch of images on CUDA. Shape (batch_size, channels, height, width).
    min_m (int): Minimum value for m.
    max_m (int): Maximum value for m.
    min_n (int): Minimum value for n.
    max_n (int): Maximum value for n.

    Returns:
    torch.Tensor: Images with randomly blocked boxes.
    """
    batch_size, channels, height, width = images.shape
    
    # Randomly generate m and n
    m = random.randint(min_m, max_m)
    n = random.randint(min_n, max_n)
    
    box_height = height // m
    box_width = width // n
    
    # Create a grid of m x n boxes
    mask = torch.ones((batch_size, 1, height, width), device=images.device)
    
    for i in range(m):
        for j in range(n):
            if torch.rand(1).item() > 0.5:  # Randomly decide whether to block this box
                mask[:, :, i*box_height:(i+1)*box_height, j*box_width:(j+1)*box_width] = 0
    
    # Apply the mask to the images
    masked_images = images * mask
    
    return masked_images

class Augmentation:
    def __init__(self):
        pass

    # def augment_rgb(self, batch):
    #         rgb = batch['positive']['rgb']
    #         depth_euclid = batch['positive']['depth_euclidean']
    #         reshade = batch['positive']['reshading']

            # # # unnormalize before applying augs
            # augmented_rgb = (rgb + 1. ) / 2.
    #         # augmented_rgb = rgb
            
    #         # augmented_rgb = color_quant(augmented_rgb, severity=4)
    #         # augmented_rgb = iso_noise(augmented_rgb, severity=3)
    #         # augmented_rgb = low_light(augmented_rgb, severity=6)



    #         # if len(augmented_rgb.size())==3:
    #         #     augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
    #         #     depth_euclid = depth_euclid.cuda().unsqueeze(0)
    #         #     # reshade = reshade.cuda().unsqueeze(0)
                
    #         # augmented_rgb = fog_3d(augmented_rgb, depth_euclid)
    #         # augmented_rgb = defocus_blur_3d_random(augmented_rgb, depth_euclid) 
    #         # augmented_rgb = motionkornia_blur_3d(augmented_rgb, depth_euclid)
    #         # augmented_rgb = zoomkornia_blur_3d(augmented_rgb, depth_euclid)  # doesn't work

    #         # New 2D augmentations
            
    #         if len(augmented_rgb.size())==3:
    #             augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
            
    #         # aug = RandomGaussianNoise(mean=0., std=random.uniform(.1, .6), p=1.)
    #         # augmented_rgb = aug(augmented_rgb)
            
    #         # augmented_rgb = augmented_rgb.cpu() # black image!
    #         # aug = RandomPosterize(3, p=1.)
    #         # augmented_rgb = aug(augmented_rgb)
    #         # augmented_rgb = augmented_rgb.cuda()
        
    #         # # 2D Kornia augmentations
    #         # # color jitter
    #         # jitter = ColorJitter(0.15, 0.15, 0.15, 0.15, p=1.)
    #         # aug = jitter
    #         # augmented_rgb = aug(augmented_rgb) # black image!
        
    #         # aug = jpeg_compression
    #         # augmented_rgb = aug(augmented_rgb)  # black
        
    #         # # print("pixelate")
    #         # aug = pixelate
    #         # # augmented_rgb = aug(augmented_rgb)
        
    #         # aug = shot_noise
    #         # augmented_rgb = aug(augmented_rgb) # doesn't work

        
    #         # aug = impulse_noise
    #         # augmented_rgb = aug(augmented_rgb) # too much

        
    #         # aug = defocus_blur
    #         # augmented_rgb = aug(augmented_rgb) # black
        
    #         # aug = fog
    #         # augmented_rgb = aug(augmented_rgb) # doesn't work
        
    #         # aug = zoom_blur
    #         # augmented_rgb = aug(augmented_rgb) # doesn't work
           
    #         #normalize back
    #         augmented_rgb = (augmented_rgb-0.5) / 0.5

    #         return augmented_rgb

    def augment_rgb(self, batch):
            rgb = batch['positive']['rgb']
            depth_euclid = batch['positive']['depth_euclidean']
            reshade = batch['positive']['reshading']

            # # 2D Kornia augmentations
            # # color jitter
            # jitter = ColorJitter(0.15, 0.15, 0.15, 0.15, p=1.)

            # unnormalize before applying augs
            # augmented_rgb = (rgb + 1. ) / 2.
            augmented_rgb = rgb

            p = random.random()
            if p < 0.2:
                p = random.random()
                if p < 0.7:
                    s = random.randint(1, 6)
                    augmented_rgb = low_light(augmented_rgb, severity=s)
                else:
                    s = random.randint(1, 4)
                    augmented_rgb = iso_noise(augmented_rgb, severity=s)
                # else:
                    # s = random.randint(1, 5)
                    # augmented_rgb = color_quant(augmented_rgb, severity=s)

            # # base omni augmentations
            elif p < 0.5: #0.7:
                p = random.random()
                if p < 0.2:
                    aug = RandomSharpness(.3, p=1.)
                    augmented_rgb = aug(augmented_rgb)

                p = random.random()
                if p < 0.5:
                    aug = RandomMotionBlur((3, 7), random.uniform(10., 50.), 0.5, p=1.)
                    augmented_rgb = aug(augmented_rgb)

                p = random.random()
                if p < 0.1:
                    aug = RandomGaussianBlur((7, 7), (0.1, 2.0), p=1.)
                    augmented_rgb = aug(augmented_rgb)
                elif p < 0.4:
                    aug = RandomGaussianBlur((5, 5), (0.1, 2.0), p=1.)
                    augmented_rgb = aug(augmented_rgb)
                elif p < 0.6:
                    aug = RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.)
                    augmented_rgb = aug(augmented_rgb)

            # # New 2D augmentations
            # elif p < 0.7:
            #     try:
            #         p = random.random()
            #         #print(p)
            #         if len(augmented_rgb.size())==3:
            #             augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
            #         if p < 0.10:
            #             aug = RandomGaussianNoise(mean=0., std=random.uniform(.1, .6), p=1.)
            #             augmented_rgb = aug(augmented_rgb)
            #         elif p < 0.20:
            #             augmented_rgb = augmented_rgb.cpu()
            #             aug = RandomPosterize(3, p=1.)
            #             augmented_rgb = aug(augmented_rgb)
            #             augmented_rgb = augmented_rgb.cuda()
            #         elif p < 0.30:
            #             aug = jitter
            #             augmented_rgb = aug(augmented_rgb)
            #         elif p < 0.40:
            #             aug = jpeg_compression
            #             augmented_rgb = aug(augmented_rgb)
            #         elif p < 0.50:
            #             # print("pixelate")
            #             aug = pixelate
            #             # augmented_rgb = aug(augmented_rgb)
            #         elif p < 0.60:
            #             aug = shot_noise
            #             augmented_rgb = aug(augmented_rgb)
            #         elif p < 0.70:
            #             aug = impulse_noise
            #             augmented_rgb = aug(augmented_rgb)
            #         elif p < 0.80:
            #             aug = defocus_blur
            #             augmented_rgb = aug(augmented_rgb)
            #         elif p < 0.90:
            #             aug = fog
            #             augmented_rgb = aug(augmented_rgb)
            #         else:
            #             aug = zoom_blur
            #             augmented_rgb = aug(augmented_rgb)
            #         # print(aug)
            #     except:
            #         # print("No 2D aug!")
            #         augmented_rgb = augmented_rgb

            # New 3D augmentations
            elif p < 0.8:
                p = random.random()
                if len(augmented_rgb.size())==3:
                    augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
                    depth_euclid = depth_euclid.cuda().unsqueeze(0)
                    # reshade = reshade.cuda().unsqueeze(0)
                
                if p < 0.7:
                    # print('fog3d')
                    augmented_rgb = fog_3d(augmented_rgb, depth_euclid)
                elif p < 0.85:
                    # print('defocus3d')
                    augmented_rgb = defocus_blur_3d_random(augmented_rgb, depth_euclid) 
                else:
                    # print('motion3d')
                    augmented_rgb = motionkornia_blur_3d(augmented_rgb, depth_euclid)
            # else:
            #     # print('zoom3d')
            #     # augmented_rgb = zoomkornia_blur_3d(augmented_rgb, depth_euclid)

            else: # Random block mask
                augmented_rgb = random_block_mask(augmented_rgb, min_m=4, max_m=20, min_n=4, max_n=20)  # Mask images with random grid size

            # normalize back
            # augmented_rgb = (augmented_rgb-0.5) / 0.5
            return augmented_rgb
    
    def augment_rgb_val(self, batch):
            rgb = batch['positive']['rgb']
            depth_euclid = batch['positive']['depth_euclidean']
            reshade = batch['positive']['reshading']

            augmented_rgb = rgb

            p = random.random()
            if p < 0.3:
                s = random.randint(1, 6)
                augmented_rgb = low_light(augmented_rgb, severity=s)

            elif p < 0.7:
                p = random.random()
                if len(augmented_rgb.size())==3:
                    augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
                    depth_euclid = depth_euclid.cuda().unsqueeze(0)
                    # reshade = reshade.cuda().unsqueeze(0)
            
                # print('fog3d')
                augmented_rgb = fog_3d_val(augmented_rgb, depth_euclid)
                
            return augmented_rgb
    
    def crop_augmentation(self, batch, tasks, fixed_size=[None, None]):
    
        original_h = fixed_size[0]
        original_w = fixed_size[1]

        aspect_ratio = original_w / original_h

        p = random.random()
        resize_method = 'centercrop' if p < 0.7 else 'randomcrop'

        img_sizes = [270, 432, 540, 648, 720, 810, 864, 900]
        h = random.choice(img_sizes)
        w = int(h * aspect_ratio)

        for task in tasks:
            if len(batch[task].shape) == 3:
                batch[task] = batch[task].unsqueeze(axis=0)
            elif len(batch[task].shape) == 2:
                batch[task] = batch[task].unsqueeze(axis=0).unsqueeze(axis=1)


            crop_h = min(h, original_h)
            crop_w = min(w, original_w)

            original_dtype = batch[task].dtype  # Save the original data type

            if resize_method == 'centercrop':
                centercrop = CenterCrop((crop_h, crop_w), p=1.)
                
                # Temporarily convert the tensor to float32 for the augmentation
                batch[task] = batch[task].to(torch.float32)
                
                # Apply the centercrop or any other Kornia augmentation
                batch[task] = centercrop(batch[task])
                
                # Convert the tensor back to its original data type
                batch[task] = batch[task].to(original_dtype)

            elif resize_method == 'randomcrop':
                min_x = random.randint(0, original_w - crop_w)
                min_y = random.randint(0, original_h - crop_h)

                batch[task] = batch[task][:, :, min_y:min_y + crop_h, min_x:min_x + crop_w]

            # Resize to original dimensions
            if original_dtype == torch.int64:
                batch[task] = batch[task].to(torch.float32)

            batch[task] = F.interpolate(batch[task], (original_h, original_w), mode='bilinear' if task == 'rgb' else 'nearest')
        
            if task == 'gt':
                batch[task] = batch[task].squeeze(axis=0)

            # Convert the tensor back to its original data type
            if original_dtype == torch.int64:
                batch[task] = batch[task].to(original_dtype)

        return batch
