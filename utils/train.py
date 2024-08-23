import os
import pprint
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset

from utils.refocus_augmentation import RefocusImageAugmentation
from utils.augmentation import Augmentation
from utils.evaluation.metrics import dice_scores, normalized_surface_distances

from utils.init_func import group_weight
from utils.init_func import configure_optimizers
from utils.lr_policy import WarmUpPolyLR
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from tensorboardX import SummaryWriter
# from val_mm import evaluate, evaluate_msf
from importlib import import_module
import datetime

import matplotlib.pyplot as plt

from utils.metrics_new import Metrics
from torchvision import transforms as T
from matplotlib.colors import ListedColormap
import pathlib
# from eval import evaluate_mid


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--val_amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--local-rank", default=0)
# parser.add_argument('--save_path', '-p', default=None)

# os.environ['MASTER_PORT'] = '169710'
torch.set_float32_matmul_precision("high")
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.automatic_dynamic_shapes = False


def is_eval(epoch, config):
    return epoch > int(config.checkpoint_start_epoch) or epoch == 1 or epoch % 1 == 0

@torch.no_grad()
def evaluate(model, dataloader, config, device, engine, save_dir=None, sliding=False, depth_model=None):
    print("Evaluating...")
    model.eval()
    n_classes = config.num_classes
    metrics = Metrics(n_classes, config.background, device)

    dice_tools = []
    nsds = []
    val_loss_list = []

    for idx, minibatch in enumerate(dataloader):
        if ((idx + 1) % int(len(dataloader) * 0.5) == 0 or idx == 0) and (
            (engine.distributed and (engine.local_rank == 0))
            or (not engine.distributed)
        ):
            print(f"Validation Iter: {idx + 1} / {len(dataloader)}")
        images = minibatch["data"]
        labels = minibatch["label"]
        # modal_xs = minibatch["modal_x"]
        modal_xs = predict_depth(depth_model, images)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if len(modal_xs.shape) == 3:
            modal_xs = modal_xs.unsqueeze(0)
        if len(labels.shape) == 2:
            labels = labels.unsqueeze(0)
        # print(images.shape,labels.shape)
        transform = T.Resize((config.image_height, config.image_width))
        images = transform(images)
        modal_xs = transform(modal_xs)
        labels = transform(labels)
        images = images.to(device)
        modal_xs = modal_xs.to(device)
        labels = labels.to(device)
        
        # Add 3DCC data augmentation
        aug = Augmentation()

        for i in range(minibatch["data"].shape[0]):
            batch = {'positive': {'rgb': None, 'depth_euclidean': None, 'gt': None, 'reshading': None}}
            
            batch['positive']['rgb'] = images[i]
            batch['positive']['depth_euclidean'] = modal_xs[i]
            batch['positive']['gt'] = labels[i]

            # Crop
            tasks = ['rgb','depth_euclidean', 'gt']
            batch['positive'] = aug.crop_augmentation(batch['positive'], tasks, fixed_size=[config.image_height, config.image_width])

            augmented_rgb = aug.augment_rgb(batch)

            augmented_rgb = augmented_rgb.squeeze(0)

            images[i] = augmented_rgb

        # calculate depth for augmented images
        images = images.cpu()
        modal_xs = predict_depth(depth_model, images)
        modal_xs = transform(modal_xs)
        modal_xs = modal_xs.to(device)

        # Add validation loss
        val_loss = model(images.to(device), modal_xs, labels)
        val_loss = val_loss.detach().cpu().numpy()
        val_loss_list.append(val_loss)

        images = [images.to(device), modal_xs.to(device)]

        # if sliding:
        #     preds = slide_inference(model, images, modal_xs, config).softmax(dim=1)
        # else:
        preds = model(images[0], images[1]).softmax(dim=1)
        # print(preds.shape,labels.shape)
        B, H, W = labels.shape
        metrics.update(preds, labels)

        # DSC and NSD metrics from SeStrongC

        # Convert tensors to numpy arrays
        preds_np = preds.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Convert to boolean arrays
        preds_bool = preds_np == 1  # Assuming class 1 is the object of interest
        labels_bool = labels_np == 1  # Same assumption for labels

        for i in range(preds_bool.shape[0]):
            dice_tool = dice_scores(preds_bool[i], labels_bool[i])
            dice_tools.append(dice_tool)
        
            nsd = normalized_surface_distances(preds_bool[i], labels_bool[i], tau=5) # tau default is 5, arg.tau
            nsds.append(nsd)

        # for i in range(B):
        #     metrics.update(preds[i].unsqueeze(0), labels[i].unsqueeze(0))
        # metrics.update(preds, labels)

        if save_dir is not None:
            palette = [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
            ]
            palette = np.array(palette, dtype=np.uint8)
            cmap = ListedColormap(palette)
            names = (
                minibatch["fn"][0]
                .replace(".jpg", "")
                .replace(".png", "")
                .replace("datasets/", "")
            )
            save_name = save_dir + "/" + names + "_pred.png"
            pathlib.Path(save_name).parent.mkdir(parents=True, exist_ok=True)
            preds = preds.argmax(dim=1).cpu().squeeze().numpy().astype(np.uint8)
            if config.dataset_name in ["KITTI-360", "EventScape"]:
                preds = palette[preds]
                plt.imsave(save_name, preds)
            elif config.dataset_name in ["NYUDepthv2", "SUNRGBD"]:
                palette = np.load("./utils/nyucmap.npy")
                preds = palette[preds]
                plt.imsave(save_name, preds)
            elif config.dataset_name in ["MFNet"]:
                palette = np.array(
                    [
                        [0, 0, 0],
                        [64, 0, 128],
                        [64, 64, 0],
                        [0, 128, 192],
                        [0, 0, 192],
                        [128, 128, 0],
                        [64, 64, 128],
                        [192, 128, 128],
                        [192, 64, 0],
                    ],
                    dtype=np.uint8,
                )
                preds = palette[preds]
                plt.imsave(save_name, preds)
            else:
                assert 1 == 2

        if idx >= 10:
            break

    # ious, miou = metrics.compute_iou()
    # acc, macc = metrics.compute_pixel_acc()
    # f1, mf1 = metrics.compute_f1()
    if engine.distributed:
        all_metrics = [None for _ in range(engine.world_size)]
        # all_predictions = Metrics(n_classes, config.background, device)
        torch.distributed.all_gather_object(all_metrics, metrics)  # list of lists
    else:
        all_metrics = metrics

    mdsc = sum(dice_tools) / len(dice_tools) if dice_tools else float('nan')
    mnsd = sum(nsds) / len(nsds) if nsds else float('nan')

    # Convert the list to a NumPy array
    val_loss_list_np = np.array(val_loss_list)
    # Calculate the mean, ignoring NaN values
    mval_loss = np.nanmean(val_loss_list_np)

    return all_metrics, mdsc, mnsd, mval_loss

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import cv2

def build_depth_estimation_model(device, encoder='vitl'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'./checkpoints/pretrained/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    return model

def predict_depth(depth_model, images):
    # image shape (B, 3, H, W)
    depths = torch.zeros_like(images)
    H, W = images.shape[2], images.shape[3]
    for batch_idx in range(images.shape[0]):
        image = images[batch_idx].numpy().transpose(1, 2, 0) # BGR
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR- > RGB
        # image = cv2.resize(image, (1920, 1080))
        # plt.imshow(image.astype(np.uint8))
        # plt.show()
        depth = depth_model.infer_image(image)
        depth = (1 - (depth - depth.min()) / (depth.max() - depth.min()))
        depth = cv2.merge([depth, depth, depth])
        depths[batch_idx] = torch.from_numpy(depth).permute(2, 0, 1)
    # plt.imshow(depths[0].permute(1, 2, 0).numpy())
    # plt.show()
    # breakpoint()
    return depths # tensor shape (B, 3, H, W)


class gpu_timer:
    def __init__(self, beta=0.6) -> None:
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            print("Use start() before stop(). ")
        torch.cuda.synchronize()
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = (
        True  # train speed is slower after enabling this opts.
    )

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True, warn_only=True)

# Function to save metrics plot
def save_metrics_plot(epochs, miou_values, mdsc_values, mnsd_values, loss_values, val_loss_values, learning_rates, filename):
    plt.figure(figsize=(15, 10))
    
    # Plot mIoU
    plt.subplot(2, 3, 1)
    plt.plot(epochs, miou_values, label='mIoU', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.title('mIoU over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot mIoU
    plt.subplot(2, 3, 2)
    plt.plot(epochs, mdsc_values, label='mDSC', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('mDSC')
    plt.title('mDSC over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot mIoU
    plt.subplot(2, 3, 3)
    plt.plot(epochs, mnsd_values, label='mNSD', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('mNSD')
    plt.title('mNSD over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(2, 3, 4)
    plt.plot(epochs, loss_values, label='Training Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_loss_values, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot Learning Rate
    plt.subplot(2, 3, 6)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)  # Save the plot to a file
# Function to save metrics plot
def save_metrics_plot(epochs, miou_values, mdsc_values, mnsd_values, loss_values, val_loss_values, learning_rates, filename):
    plt.figure(figsize=(15, 10))
    
    # Plot mIoU
    plt.subplot(2, 3, 1)
    plt.plot(epochs, miou_values, label='mIoU', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.title('mIoU over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot mIoU
    plt.subplot(2, 3, 2)
    plt.plot(epochs, mdsc_values, label='mDSC', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('mDSC')
    plt.title('mDSC over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot mIoU
    plt.subplot(2, 3, 3)
    plt.plot(epochs, mnsd_values, label='mNSD', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('mNSD')
    plt.title('mNSD over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(2, 3, 4)
    plt.plot(epochs, loss_values, label='Training Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_loss_values, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot Learning Rate
    plt.subplot(2, 3, 6)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)  # Save the plot to a file

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    config = getattr(import_module(args.config), "C")
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    if args.use_seed:
        set_seed(config.seed)
        logger.info(f"set seed {config.seed}")
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True 
        logger.info("use random seed")

    # assert not (args.compile and args.syncbn), "syncbn is not supported in compile mode"
    if not args.compile and args.compile_mode != "default":
        logger.warning(
            "compile_mode is only valid when compile is enabled, ignoring compile_mode"
        )

    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

    if args.gpus == 2:
        if args.mst and args.compile and args.compile_mode == "reduce-overhead":
            val_dl_factor = 0.25
        elif args.mst and not args.val_amp:
            val_dl_factor = 1.5
        elif args.mst and args.val_amp:
            val_dl_factor = 1.3
        else:
            val_dl_factor = 2
    elif args.gpus == 4:
        if args.mst and args.compile and args.compile_mode == "reduce-overhead":
            val_dl_factor = 0.25
        elif args.mst and not args.val_amp:
            val_dl_factor = 1.5
        elif args.mst and args.val_amp:
            val_dl_factor = 0.6
        else:
            val_dl_factor = 2
    else:
        val_dl_factor = 1.5

    val_dl_factor = 1 # TODO: remove this line

    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=1,
        # val_batch_size=int(config.batch_size * val_dl_factor) if config.dataset_name!="SUNRGBD" else int(args.gpus),
    )
    logger.info(f"val dataset len:{len(val_loader)*int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(
            time.strftime("%b%d_%d-%H-%M", time.localtime())
        )
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n" + pp.pformat(config))

    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)

    if args.syncbn:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    # weight=torch.load('checkpoints/NYUv2_DFormer_Large.pth')['model']
    # w_list=list(weight.keys())
    # # for k in w_list:
    # #     weight[k[7:]] = weight[k]
    # print('load model')
    # model.load_state_dict(weight)

    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    # params_list = configure_optimizers(model, base_lr, config.weight_decay)

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch,
    )
    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=False,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        depth_model = build_depth_estimation_model(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()

    logger.info("begin trainning:")
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "class_names": config.class_names,
    }
    # val_pre = ValPre()
    # val_dataset = RGBXDataset(data_setting, 'val', val_pre)
    # test_loader, test_sampler = get_test_loader(engine, RGBXDataset,config)
    all_dev = [0]
    # segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, None,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, config,args.verbose, args.save_path,args.show_image)
    uncompiled_model = model
    if args.compile:
        compiled_model = torch.compile(
            model, backend="inductor", mode=args.compile_mode
        )
    else:
        compiled_model = model
    miou, best_miou = 0.0, 0.0
    train_timer = gpu_timer()
    eval_timer = gpu_timer()

    epochs = []
    miou_values = []
    mdsc_values = []
    mnsd_values = []
    loss_values = []
    val_loss_values = []
    learning_rates = []

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model = compiled_model
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        # bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
        # pbar = tqdm(
        #     range(config.niters_per_epoch),
        #     file=sys.stdout,
        #     bar_format=bar_format,
        #     # range(5),
        #     # file=sys.stdout,
        #     # bar_format=bar_format,
        # )
        dataloader = iter(train_loader)

        sum_loss = 0
        i = 0
        train_timer.start()
        for idx in range(config.niters_per_epoch):
            engine.update_iteration(epoch, idx)

            # minibatch = dataloader.next()
            minibatch = next(dataloader)
            imgs = minibatch["data"]
            gts = minibatch["label"]
            # modal_xs = minibatch["modal_x"]
            modal_xs = predict_depth(depth_model, imgs)

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            # Add 3DCC data augmentation
            aug = Augmentation()

            for i in range(minibatch["data"].shape[0]):
                batch = {'positive': {'rgb': None, 'depth_euclidean': None, 'gt': None, 'reshading': None}}
            
                batch['positive']['rgb'] = imgs[i]
                batch['positive']['depth_euclidean'] = modal_xs[i]
                batch['positive']['gt'] = gts[i]

                # augmentation
                # Crop
                tasks = ['rgb','depth_euclidean', 'gt']
                batch['positive'] = aug.crop_augmentation(batch['positive'],tasks, fixed_size=[config.image_height, config.image_width])

                # # Visulization
                # # Original Image (before augmentation)
                # original_rgb = imgs[i].permute(1, 2, 0).cpu().numpy()  # Convert to HxWxC

                # # If original image is in BGR format, convert to RGB
                # original_rgb = original_rgb[..., [2, 1, 0]]  # Swap BGR to RGB

                # # Normalize original image to [0, 1] range
                # original_rgb = (original_rgb - original_rgb.min()) / (original_rgb.max() - original_rgb.min())

                # Augmentation
                augmented_rgb = aug.augment_rgb(batch)
                augmented_rgb = augmented_rgb.squeeze(0)
                imgs[i] = augmented_rgb

                # # Visulization
                # # Convert to NumPy if necessary
                # augmented_rgb_vis = augmented_rgb.permute(1, 2, 0).cpu().numpy()  # Change from CxHxW to HxWxC

                # # If the image is in BGR, convert it to RGB
                # augmented_rgb_vis = augmented_rgb_vis[..., [2, 1, 0]] # Swap BGR to RGB

                # # Normalize to [0, 1]
                # augmented_rgb_vis = (augmented_rgb_vis - augmented_rgb_vis.min()) / (augmented_rgb_vis.max() - augmented_rgb_vis.min())

                # # Display the original and augmented images side by side
                # plt.figure(figsize=(10, 5))

                # # Original Image
                # plt.subplot(1, 2, 1)
                # plt.imshow(original_rgb)
                # plt.title('Original Image')
                # plt.axis('off')

                # # Augmented Image
                # plt.subplot(1, 2, 2)
                # plt.imshow(augmented_rgb_vis)
                # plt.title('Augmented Image')
                # plt.axis('off')

                # plt.show()

            if args.amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # print(imgs.shape) # (B, D, H, W)
                    # print(modal_xs.shape) # (B, 1, H, W)
                    # print(gts.shape) # (B, H, W)
                    loss = model(imgs, modal_xs, gts)
            else:
                # print(imgs.shape) # (B, D, H, W)
                # print(modal_xs.shape) # (B, 1, H, W)
                # print(gts.shape) # (B, H, W)
                imgs = imgs.cpu()
                modal_xs = predict_depth(depth_model, imgs)
                modal_xs = modal_xs.cuda(non_blocking=True)
                imgs = imgs.cuda(non_blocking=True)

                loss = model(imgs, modal_xs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            if args.amp:
                # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                optimizer.zero_grad(
                    set_to_none=True
                )  # TODO: check if set_to_none=True impact the performance
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if not args.amp:
                if epoch == 1:
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            logger.warning(f"{name} has no grad, please check")

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = (
                    "Epoch {}/{}".format(epoch, config.nepochs)
                    + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                    + " lr=%.4e" % lr
                    + " loss=%.4f total_loss=%.4f"
                    % (reduce_loss.item(), (sum_loss / (idx + 1)))
                )

            else:
                sum_loss += loss
                print_str = (
                    f"Epoch {epoch}/{config.nepochs} "
                    + f"Iter {idx + 1}/{config.niters_per_epoch}: "
                    + f"lr={lr:.4e} loss={loss:.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                )

            if ((idx + 1) % int((config.niters_per_epoch) * 0.1) == 0 or idx == 0) and (
                (engine.distributed and (engine.local_rank == 0))
                or (not engine.distributed)
            ):
                print(print_str)

            del loss
            # pbar.set_description(print_str, refresh=False)
        logger.info(print_str)
        train_timer.stop()

        # if (engine.distributed and (engine.local_rank == 0)) or (
        #     not engine.distributed
        # ):
        #     tb.add_scalar("train_loss", sum_loss / len(pbar), epoch)

        if is_eval(epoch, config):
            eval_timer.start()
            torch.cuda.empty_cache()
            # if args.compile and args.mst and (not args.sliding):
            #     model = uncompiled_model
            # TODO: FIX this
            if engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.val_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            all_metrics, mdsc, mnsd, mval_loss = evaluate(
                                model,
                                val_loader,
                                config,
                                device,
                                engine,
                                sliding=args.sliding,
                            )
                    else:
                        all_metrics, mdsc, mnsd, mval_loss = evaluate(
                            model,
                            val_loader,
                            config,
                            device,
                            engine,
                            sliding=args.sliding,
                        )
            
                    if engine.local_rank == 0:
                        metric = all_metrics[0]
                        for other_metric in all_metrics[1:]:
                            metric.update_hist(other_metric.hist)
                        ious, miou = metric.compute_iou()
                        acc, macc = metric.compute_pixel_acc()
                        f1, mf1 = metric.compute_f1()
                        if miou > best_miou:
                            best_miou = miou
                            engine.save_and_link_checkpoint(
                                config.log_dir,
                                config.log_dir,
                                config.log_dir_link,
                                infor="_miou_" + str(miou),
                                metric=miou,
                            )
                        print("miou", miou, "best", best_miou)
            elif not engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.val_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            metric, mdsc, mnsd, mval_loss = evaluate(
                                model,
                                val_loader,
                                config,
                                device,
                                engine,
                                sliding=args.sliding,
                                depth_model=depth_model,
                            )
                    else:
                        metric, mdsc, mnsd, mval_loss = evaluate(
                            model,
                            val_loader,
                            config,
                            device,
                            engine,
                            sliding=args.sliding,
                        )

                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    # print('miou',miou)
                # print('acc, macc, f1, mf1, ious, miou',acc, macc, f1, mf1, ious, miou)
                # print('miou',miou)
                if miou > best_miou:
                    best_miou = miou
                    engine.save_and_link_checkpoint(
                        config.log_dir,
                        config.log_dir,
                        config.log_dir_link,
                        infor="_miou_" + str(miou),
                        metric=miou,
                    )
                print("miou", miou, "best", best_miou)
            logger.info(
                f"Epoch {epoch} validation result: mIoU {miou}, best mIoU {best_miou}"
            )
            eval_timer.stop()

            if epoch != 1: 
                # Save plot every 10 epochs
                save_metrics_plot(epochs, miou_values, mdsc_values, mnsd_values, loss_values, val_loss_values, learning_rates, filename='metrics_plot.png')

        eval_count = 0
        for i in range(engine.state.epoch + 1, config.nepochs + 1):
            if is_eval(i, config):
                eval_count += 1
        left_time = (
            train_timer.mean_time * (config.nepochs - engine.state.epoch)
            + eval_timer.mean_time * eval_count
        )
        eta = (
            datetime.datetime.now() + datetime.timedelta(seconds=left_time)
        ).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"Avg train time: {train_timer.mean_time:.2f}s, avg eval time: {eval_timer.mean_time:.2f}s, left eval count: {eval_count}, ETA: {eta}"
        )

        epochs.append(epoch)
        miou_values.append(miou)
        mdsc_values.append(mdsc)
        mnsd_values.append(mnsd)
        loss_values.append(sum_loss.detach().cpu().numpy()/ (idx + 1))
        val_loss_values.append(mval_loss)
        learning_rates.append(lr)
    
    # Convert lists to numpy arrays
    epochs_np = np.array(epochs)
    loss_np = np.array(loss_values)
    mval_loss_np = np.array(val_loss_values)
    miou_np = np.array(miou_values)
    mdsc_np = np.array(mdsc_values)
    mnsd_np = np.array(mnsd_values)
    learning_rate_np = np.array(learning_rates)

    # Save data
    np.savez('metrics.npz', epochs=epochs_np, loss=loss_np, val_loss = mval_loss_np, miou=miou_np, mdsc = mdsc_np, mnsd = mdsc_np, learning_rate=learning_rate_np)

    # # Load data
    # data = np.load('metrics.npz')
    # epochs = data['epochs']
    # loss_values = data['loss']
    # miou_values = data['miou']
    # learning_rates = data['learning_rate']
