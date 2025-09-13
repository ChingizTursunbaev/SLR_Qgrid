import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import random
from pathlib import Path
from functools import partial

from timm.utils import ModelEma

# --- THIS IS THE FINAL, CORRECTED PART ---
# Use absolute imports from the 'slr' package, which is the root of your code
from slr.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from slr.engine import train_one_epoch, evaluate
from slr.utils import NativeScalerWithGradNormCount as NativeScaler
from slr import utils
import contextlib

from slr.models.multi_modal_model import MultiModalMamba
from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.datasets import video_transforms, volume_transforms
# --- END OF CORRECTION ---

def get_args_parser():
    parser = argparse.ArgumentParser('Mamba SLR training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--headdim', default=64, type=int)
    parser.add_argument('--depth', default=12, type=int, help="Depth of the image/keypoint encoder branch.")
    parser.add_argument('--qgrid_depth', default=18, type=int, help="Depth of the Qgrid encoder branch.")
    parser.add_argument('--expand', default=2, type=int)

    parser.add_argument('--head_drop_rate', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--model_ema_eval', action='store_true', default=False, help='Using ema to eval during training.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--prefix', default='', type=str, help='prefix for image data', required=True)
    parser.add_argument('--gloss_dict_path', default='', type=str, required=True)
    parser.add_argument('--meta_dir_path', default='', type=str, required=True)
    parser.add_argument('--qgrid_prefix', type=str, required=True, help='Path to Qgrid data files directory')
    parser.add_argument('--kp_path', default='', type=str, required=True, help='Path to keypoints .pkl file')
    
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true', default=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--test_best', action='store_true', help='Whether to test the best model')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    
    # bf16 and no_amp parameters
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--no-bf16', action='store_false', dest='bf16')
    parser.add_argument('--no_amp', action='store_true', default=False, help="Disable AMP")
    
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    gloss_dict = np.load(args.gloss_dict_path, allow_pickle=True).item()
    num_classes = len(gloss_dict)

    # --- DATASET & DATALOADER ---
    transform_train = video_transforms.Compose([
        volume_transforms.ClipToTensor(),
        partial(video_transforms.random_resized_crop, target_height=args.input_size, target_width=args.input_size, scale=(0.9, 1.0)),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = video_transforms.Compose([
        volume_transforms.ClipToTensor(),
        video_transforms.Resize(args.input_size),
        video_transforms.CenterCrop(args.input_size),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train = MultiModalPhoenixDataset(
        image_prefix=args.prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir_path,
        gloss_dict_path=args.gloss_dict_path,
        split='train',
        transforms=transform_train,
    )

    if not args.disable_eval_during_finetuning:
        dataset_val = MultiModalPhoenixDataset(
            image_prefix=args.prefix,
            qgrid_prefix=args.qgrid_prefix,
            kp_path=args.kp_path,
            meta_dir_path=args.meta_dir_path,
            gloss_dict_path=args.gloss_dict_path,
            split='dev',
            transforms=transform_val,
        )
    else:
        dataset_val = None

    dataset_test = MultiModalPhoenixDataset(
        image_prefix=args.prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir_path,
        gloss_dict_path=args.gloss_dict_path,
        split='test',
        transforms=transform_val,
    )

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if args.dist_eval else torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False) if args.dist_eval else torch.utils.data.SequentialSampler(dataset_test)

    log_writer = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        collate_fn=multi_modal_collate_fn, persistent_workers=args.num_workers > 0
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size * 2,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
        collate_fn=multi_modal_collate_fn, persistent_workers=args.num_workers > 0
    ) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test, batch_size=args.batch_size * 2,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
        collate_fn=multi_modal_collate_fn, persistent_workers=args.num_workers > 0
    )

    # --- MODEL ---
    base_cfg = {
        'd_model': args.d_model, 'd_intermediate': args.d_model * 4,
        'drop_path_rate': args.drop_path, 'head_drop_rate': args.head_drop_rate,
        'ssm_cfg': {'spatial': {'expand': args.expand, 'headdim': args.headdim}, 'temporal': {'expand': args.expand, 'headdim': args.headdim}},
        'attn_cfg': {'spatial': {'num_heads': args.d_model // args.headdim}, 'temporal': {}},
        'attn_layer_idx': {'spatial': [], 'temporal': []},
        'rms_norm': False, 'fused_add_norm': True, 'residual_in_fp32': True
    }
    
    img_cfg = base_cfg.copy()
    img_cfg.update({'img_size': args.input_size, 'patch_size': args.patch_size, 'n_layer': args.depth, 'channels': 3})
    
    qgrid_cfg = base_cfg.copy()
    qgrid_cfg.update({'img_size': 121, 'patch_size': 11, 'n_layer': args.qgrid_depth, 'channels': 2})

    kp_cfg = { 'input_dim': 121*2, 'model_dim': args.d_model }
    fusion_cfg = { 'embed_dim': args.d_model, 'num_heads': 8, 'dropout': 0.1 }

    model = MultiModalMamba(img_cfg=img_cfg, qgrid_cfg=qgrid_cfg, kp_cfg=kp_cfg, num_classes=num_classes, fusion_cfg=fusion_cfg)
    model.to(device)
    
    model_ema = ModelEma(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='') if args.model_ema else None
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)

    # --- OPTIMIZER, SCHEDULER, LOSS ---
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, len(data_loader_train) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr)
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader_train) // args.update_freq)

    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print("Performing evaluation only...")
        test_stats = evaluate(data_loader_test, model, device, contextlib.nullcontext(), gloss_dict)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['wer']:.2f}% WER")
        return

    # --- TRAINING LOOP ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_wer = 100.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            contextlib.nullcontext(), args.clip_grad, model_ema, log_writer,
            start_steps=epoch * (len(data_loader_train) // args.update_freq),
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=(len(data_loader_train) // args.update_freq),
            update_freq=args.update_freq
        )

        if args.output_dir:
            utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, model_name='latest', model_ema=model_ema)

        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, contextlib.nullcontext(), gloss_dict)
            print(f"WER on the validation set: {val_stats['wer']:.2f}%")

            if val_stats['wer'] < best_wer:
                best_wer = val_stats['wer']
                if args.output_dir:
                    utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, model_name='best', model_ema=model_ema)
            
            print(f'Best WER: {best_wer:.2f}%')

            if log_writer is not None:
                log_writer.update(val_wer=val_stats['wer'], head="perf", step=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.test_best:
        print("--- Testing best model ---")
        utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema, model_name='best')
        evaluate(data_loader_test, model, device, contextlib.nullcontext(), gloss_dict)

if __name__ == '__main__':
    opts, ds_init = get_args_parser()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if opts.bf16:
        opts.no_amp = True
    main(opts, ds_init)
























# import argparse
# import datetime
# import numpy as np
# import time
# import torch
# import torch.backends.cudnn as cudnn
# import json
# import os
# import random
# from pathlib import Path

# from timm.utils import ModelEma
# from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

# # [MODIFIED] --- Import the new function name 'evaluate' ---
# from engine import train_one_epoch, evaluate
# # ---

# from utils import NativeScalerWithGradNormCount as NativeScaler
# import utils
# import contextlib

# from models.multi_modal_model import MultiModalMamba
# from datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
# from datasets import video_transforms, volume_transforms

# def get_args_parser():
#     parser = argparse.ArgumentParser('Mamba SLR training and evaluation script', add_help=False)
#     parser.add_argument('--batch_size', default=1, type=int)
#     parser.add_argument('--epochs', default=100, type=int)
#     parser.add_argument('--update_freq', default=1, type=int)
#     parser.add_argument('--save_ckpt_freq', default=10, type=int)

#     # Model parameters
#     parser.add_argument('--input_size', default=224, type=int, help='videos input size')
#     parser.add_argument('--patch_size', default=16, type=int)
#     parser.add_argument('--d_model', default=512, type=int)
#     parser.add_argument('--headdim', default=64, type=int)
#     parser.add_argument('--depth', default=12, type=int, help="Depth of the image/keypoint encoder branch.")
#     parser.add_argument('--qgrid_depth', default=18, type=int, help="Depth of the Qgrid encoder branch.")
#     parser.add_argument('--expand', default=2, type=int)

#     parser.add_argument('--head_drop_rate', type=float, default=0.1, metavar='PCT')
#     parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')

#     parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
#     parser.add_argument('--model_ema', action='store_true', default=False)
#     parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
#     parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
#     parser.add_argument('--model_ema_eval', action='store_true', default=False, help='Using ema to eval during training.')

#     # Optimizer parameters
#     parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
#     parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
#     parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
#     parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
#     parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
#     parser.add_argument('--weight_decay', type=float, default=0.05)
#     parser.add_argument('--weight_decay_end', type=float, default=None)

#     parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
#     parser.add_argument('--layer_decay', type=float, default=0.75)
#     parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR')
#     parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')

#     parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
#     parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')

#     # Finetuning params
#     parser.add_argument('--finetune', default='', help='finetune from checkpoint')

#     # Dataset parameters
#     parser.add_argument('--prefix', default='', type=str, help='prefix for image data', required=True)
#     parser.add_argument('--gloss_dict_path', default='', type=str, required=True)
#     parser.add_argument('--meta_dir_path', default='', type=str, required=True)
#     parser.add_argument('--qgrid_prefix', type=str, required=True, help='Path to Qgrid data files directory')
#     parser.add_argument('--kp_path', default='', type=str, required=True, help='Path to keypoints .pkl file')
    
#     parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
#     parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
#     parser.add_argument('--device', default='cuda', help='device to use for training / testing')
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--resume', default='', help='resume from checkpoint')
#     parser.add_argument('--auto_resume', action='store_true')
#     parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
#     parser.set_defaults(auto_resume=True)

#     parser.add_argument('--save_ckpt', action='store_true', default=True)
#     parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
#     parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
#     parser.add_argument('--test_best', action='store_true', help='Whether to test the best model')
#     parser.add_argument('--dist_eval', action='store_true', default=False)
#     parser.add_argument('--num_workers', default=8, type=int)
#     parser.add_argument('--pin_mem', action='store_true', default=True)

#     # distributed training parameters
#     parser.add_argument('--world_size', default=1, type=int)
#     parser.add_argument('--local_rank', default=-1, type=int)
#     parser.add_argument('--dist_on_itp', action='store_true')
#     parser.add_argument('--dist_url', default='env://')
#     parser.add_argument('--enable_deepspeed', action='store_true', default=False)
#     parser.add_argument('--bf16', action='store_true', default=False)

#     known_args, _ = parser.parse_known_args()

#     if known_args.enable_deepspeed:
#         try:
#             import deepspeed
#             from deepspeed import DeepSpeedConfig
#             parser = deepspeed.add_config_arguments(parser)
#             ds_init = deepspeed.initialize
#         except:
#             print("Please 'pip install deepspeed'")
#             exit(0)
#     else:
#         ds_init = None

#     return parser.parse_args(), ds_init

# def main(args, ds_init):
#     utils.init_distributed_mode(args)

#     if ds_init is not None:
#         utils.create_ds_config(args)

#     print(args)
#     device = torch.device(args.device)

#     # fix the seed for reproducibility
#     seed = args.seed + utils.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True

#     gloss_dict = np.load(args.gloss_dict_path, allow_pickle=True).item()
#     num_classes = len(gloss_dict)

#     # --- DATASET & DATALOADER ---
#     transform_train = video_transforms.Compose([
#         video_transforms.RandomResizedCrop(args.input_size, scale=(0.9, 1.0)),
#         video_transforms.RandomHorizontalFlip(),
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     transform_val = video_transforms.Compose([
#         video_transforms.Resize(args.input_size),
#         video_transforms.CenterCrop(args.input_size),
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     dataset_train = MultiModalPhoenixDataset(
#         image_prefix=args.prefix,
#         qgrid_prefix=args.qgrid_prefix,
#         kp_path=args.kp_path,
#         meta_dir=args.meta_dir_path,
#         gloss_dict_path=args.gloss_dict_path,
#         split='train',
#         transforms=transform_train,
#     )

#     if not args.disable_eval_during_finetuning:
#         dataset_val = MultiModalPhoenixDataset(
#             image_prefix=args.prefix,
#             qgrid_prefix=args.qgrid_prefix,
#             kp_path=args.kp_path,
#             meta_dir=args.meta_dir_path,
#             gloss_dict_path=args.gloss_dict_path,
#             split='dev',
#             transforms=transform_val,
#         )
#     else:
#         dataset_val = None

#     dataset_test = MultiModalPhoenixDataset(
#         image_prefix=args.prefix,
#         qgrid_prefix=args.qgrid_prefix,
#         kp_path=args.kp_path,
#         meta_dir=args.meta_dir_path,
#         gloss_dict_path=args.gloss_dict_path,
#         split='test',
#         transforms=transform_val,
#     )

#     num_tasks = utils.get_world_size()
#     global_rank = utils.get_rank()

#     sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
#     sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if args.dist_eval else torch.utils.data.SequentialSampler(dataset_val)
#     sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False) if args.dist_eval else torch.utils.data.SequentialSampler(dataset_test)

#     log_writer = None
#     if global_rank == 0 and args.log_dir is not None:
#         os.makedirs(args.log_dir, exist_ok=True)
#         log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

#     data_loader_train = torch.utils.data.DataLoader(
#         dataset_train, sampler=sampler_train, batch_size=args.batch_size,
#         num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
#         collate_fn=multi_modal_collate_fn, persistent_workers=args.num_workers > 0
#     )

#     data_loader_val = torch.utils.data.DataLoader(
#         dataset_val, sampler=sampler_val, batch_size=args.batch_size * 2,
#         num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
#         collate_fn=multi_modal_collate_fn, persistent_workers=args.num_workers > 0
#     ) if dataset_val is not None else None

#     data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, sampler=sampler_test, batch_size=args.batch_size * 2,
#         num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
#         collate_fn=multi_modal_collate_fn, persistent_workers=args.num_workers > 0
#     )

#     # --- MODEL ---
#     base_cfg = {
#         'd_model': args.d_model, 'd_intermediate': args.d_model * 4,
#         'drop_path_rate': args.drop_path, 'head_drop_rate': args.head_drop_rate,
#         'ssm_cfg': {'spatial': {'expand': args.expand, 'headdim': args.headdim}, 'temporal': {'expand': args.expand, 'headdim': args.headdim}},
#         'attn_cfg': {'spatial': {'num_heads': args.d_model // args.headdim}, 'temporal': {}},
#         'attn_layer_idx': {'spatial': [], 'temporal': []},
#         'rms_norm': False, 'fused_add_norm': True, 'residual_in_fp32': True
#     }
    
#     img_cfg = base_cfg.copy()
#     img_cfg.update({'img_size': args.input_size, 'patch_size': args.patch_size, 'n_layer': args.depth, 'channels': 3})
    
#     qgrid_cfg = base_cfg.copy()
#     qgrid_cfg.update({'img_size': 121, 'patch_size': 11, 'n_layer': args.qgrid_depth, 'channels': 2}) # Assuming Qgrid is (T, 121, 2)

#     kp_cfg = { 'input_dim': 121*2, 'model_dim': args.d_model }
#     fusion_cfg = { 'embed_dim': args.d_model, 'num_heads': 8, 'dropout': 0.1 }

#     model = MultiModalMamba(img_cfg=img_cfg, qgrid_cfg=qgrid_cfg, kp_cfg=kp_cfg, num_classes=num_classes, fusion_cfg=fusion_cfg)
#     model.to(device)
    
#     model_ema = ModelEma(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='') if args.model_ema else None
#     model_without_ddp = model
#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print('number of params:', n_parameters)

#     # --- OPTIMIZER, SCHEDULER, LOSS ---
#     optimizer = create_optimizer(args, model_without_ddp)
#     loss_scaler = NativeScaler()
#     criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

#     lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, len(data_loader_train) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr)
#     wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader_train) // args.update_freq)

#     utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

#     if args.eval:
#         print("Performing evaluation only...")
#         test_stats = evaluate(data_loader_test, model, device, contextlib.nullcontext(), gloss_dict)
#         print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['wer']:.2f}% WER")
#         return

#     # --- TRAINING LOOP ---
#     print(f"Start training for {args.epochs} epochs")
#     start_time = time.time()
#     best_wer = 100.0

#     for epoch in range(args.start_epoch, args.epochs):
#         if args.distributed:
#             data_loader_train.sampler.set_epoch(epoch)
        
#         train_stats = train_one_epoch(
#             model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
#             contextlib.nullcontext(), args.clip_grad, model_ema, log_writer,
#             start_steps=epoch * (len(data_loader_train) // args.update_freq),
#             lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
#             num_training_steps_per_epoch=(len(data_loader_train) // args.update_freq),
#             update_freq=args.update_freq
#         )

#         if args.output_dir:
#             utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, model_name='latest', model_ema=model_ema)

#         if data_loader_val is not None:
#             val_stats = evaluate(data_loader_val, model, device, contextlib.nullcontext(), gloss_dict)
#             print(f"WER on the validation set: {val_stats['wer']:.2f}%")

#             if val_stats['wer'] < best_wer:
#                 best_wer = val_stats['wer']
#                 if args.output_dir:
#                     utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, model_name='best', model_ema=model_ema)
            
#             print(f'Best WER: {best_wer:.2f}%')

#             if log_writer is not None:
#                 log_writer.update(val_wer=val_stats['wer'], head="perf", step=epoch)

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Training time {}'.format(total_time_str))

#     if args.test_best:
#         print("--- Testing best model ---")
#         utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema, model_name='best')
#         evaluate(data_loader_test, model, device, contextlib.nullcontext(), gloss_dict)

# if __name__ == '__main__':
#     opts, ds_init = get_args_parser()
#     if opts.output_dir:
#         Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
#     main(opts, ds_init)
