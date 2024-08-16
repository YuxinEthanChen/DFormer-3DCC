from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = "SegSTRONGC"
C.dataset_path = osp.join(C.root_dir, "SegSTRONGC")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".png"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = (
    True  # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
)
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 6600
C.num_eval_imgs = 3600
C.num_classes = 2
C.class_names = ["background", "object"]

"""Image Config"""
C.background = 255
C.image_height = 512
C.image_width = 512
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
