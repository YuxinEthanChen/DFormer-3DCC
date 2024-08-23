import os
import cv2
import torch
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'../checkpoints/pretrained/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

root_dir = '/home/zijianwu/Codes/DFormer/datasets/SegSTRONGC/RGB'

imgs = sorted(os.listdir(os.path.join(root_dir)))

os.makedirs('./depth', exist_ok=True)

for img_file in tqdm(imgs):
    img = cv2.imread(os.path.join(root_dir, img_file))
    depth = model.infer_image(img) # HxW raw depth map in numpy
    # Min-max normalization
    depth = (1 - (depth - depth.min()) / (depth.max() - depth.min())) * 255
    cv2.imwrite(os.path.join('depth', img_file), depth)
    breakpoint()
    # import matplotlib.pyplot as plt
    # plt.imshow(depth)
    # plt.show()
    # breakpoint(