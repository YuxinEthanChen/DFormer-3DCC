import random
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from refocus_augmentation import RefocusImageAugmentation
from augs_3d_cc import motionkornia_blur_3d, zoomkornia_blur_3d, flash_3d, fog_3d, defocus_blur_3d_random
from augs_2d_cc import jpeg_compression, pixelate, shot_noise, impulse_noise, defocus_blur, fog, zoom_blur

import pdb
try:
    from kornia.augmentation import *
except:
    print("Error importing kornia augmentation")


class Augmentation:
    def __init__(self):
        pass

    def augment_rgb(self, batch):
            rgb = batch['positive']['rgb']
            depth_euclid = batch['positive']['depth_euclidean']
            reshade = batch['positive']['reshading']

            # 2D Kornia augmentations
            # color jitter
            jitter = ColorJitter(0.15, 0.15, 0.15, 0.15, p=1.)

            # unnormalize before applying augs
            augmented_rgb = (rgb + 1. ) / 2.
            
            p = random.random()
            # base omni augmentations
            if p < 0.4: #0.7:
                p = random.random()
                if p < 0.5:
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

            # New 2D augmentations
            elif p < 0.7:
                try:
                    p = random.random()
                    #print(p)
                    if len(augmented_rgb.size())==3:
                        augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
                    if p < 0.10:
                        aug = RandomGaussianNoise(mean=0., std=random.uniform(.1, .6), p=1.)
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.20:
                        augmented_rgb = augmented_rgb.cpu()
                        aug = RandomPosterize(3, p=1.)
                        augmented_rgb = aug(augmented_rgb)
                        augmented_rgb = augmented_rgb.cuda()
                    elif p < 0.30:
                        aug = jitter
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.40:
                        aug = jpeg_compression
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.50:
                        # print("pixelate")
                        aug = pixelate
                        # augmented_rgb = aug(augmented_rgb)
                    elif p < 0.60:
                        aug = shot_noise
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.70:
                        aug = impulse_noise
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.80:
                        aug = defocus_blur
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.90:
                        aug = fog
                        augmented_rgb = aug(augmented_rgb)
                    else:
                        aug = zoom_blur
                        augmented_rgb = aug(augmented_rgb)
                    # print(aug)
                except:
                    # print("No 2D aug!")
                    augmented_rgb = augmented_rgb

            # New 3D augmentations
            else:
                p = random.random()
                if len(augmented_rgb.size())==3:
                    augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
                    depth_euclid = depth_euclid.cuda().unsqueeze(0)
                    # reshade = reshade.cuda().unsqueeze(0)
                
                if p < 0.2:
                    # print('fog3d')
                    augmented_rgb = fog_3d(augmented_rgb, depth_euclid)
                elif p < 0.4:
                    p = p
                    # print('flash')
                    # augmented_rgb = flash_3d(augmented_rgb, reshade)
                elif p < 0.6:
                    # print('defocus3d')
                    augmented_rgb = defocus_blur_3d_random(augmented_rgb, depth_euclid) 
                elif p < 0.8:
                    # print('motion3d')
                    augmented_rgb = motionkornia_blur_3d(augmented_rgb, depth_euclid)
                else:
                    # print('zoom3d')
                    augmented_rgb = zoomkornia_blur_3d(augmented_rgb, depth_euclid)


            # normalize back
            augmented_rgb = (augmented_rgb-0.5) / 0.5
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
