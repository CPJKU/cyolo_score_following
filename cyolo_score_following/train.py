

import argparse
import json
import random
import torch
import sys

import math
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from cyolo_score_following.utils.dist_utils import *
from cyolo_score_following.utils.general import load_yaml
from cyolo_score_following.dataset import load_dataset, collate_wrapper, iterate_dataset
from cyolo_score_following.models.yolo import build_model_and_criterion as yolo_model_and_criterion
from time import gmtime, strftime
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, BatchSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter


def train(args):
    init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # uncomment line for full reproducibility (might slow down training)
    # torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True
    dump_path = None
    logging = not args.no_log

    if is_main_process() and logging:
        time_stamp = strftime("%Y%m%d_%H%M%S", gmtime()) + f"_{args.tag}"

        if not os.path.exists(args.log_root):
            os.makedirs(args.log_root)

        if not os.path.exists(args.dump_root):
            os.makedirs(args.dump_root)

        dump_path = os.path.join(args.dump_root, time_stamp)

        if not os.path.exists(dump_path):
            os.mkdir(dump_path)

    train_parameters = dict(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        dump_path=dump_path,
        augment=args.augment,
    )

    log_writer = None

    if is_main_process() and logging:
        log_dir = os.path.join(args.log_root, time_stamp)
        log_writer = SummaryWriter(log_dir=log_dir)

        text = ""
        arguments = np.sort([arg for arg in vars(args)])
        for arg in arguments:
            text += f"**{arg}:** {getattr(args, arg)}<br>"

        for key in train_parameters.keys():
            text += f"**{key}:** {train_parameters[key]}<br>"

        log_writer.add_text("run_config", text)
        log_writer.add_text("cmd", " ".join(sys.argv))

        # store the network configuration
        with open(os.path.join(dump_path, 'net_config.json'), "w") as f:
            json.dump(args.config, f)

    network, criterion = yolo_model_and_criterion(args.config)

    print(network)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        network.cuda(args.gpu)
        network = nn.parallel.DistributedDataParallel(network, device_ids=[args.gpu], output_device=args.gpu)
    else:
        network.to(device)

    train_dataset = load_dataset(args.train_set, augment=args.augment, scale_width=args.scale_width,
                                 split_file=args.train_split_file, ir_path=args.ir_path, load_audio=args.load_audio)

    val_dataset = load_dataset(args.val_set, augment=False, scale_width=args.scale_width,
                               split_file=args.val_split_file,  load_audio=args.load_audio)

    batch_size = train_parameters['batch_size']

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)
    else:
        sampler_train = RandomSampler(train_dataset)
        sampler_val = SequentialSampler(val_dataset)

    batch_sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=collate_wrapper,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size,  sampler=sampler_val, drop_last=False, collate_fn=collate_wrapper,
                            num_workers=args.num_workers, pin_memory=True)

    # as proposed in https://arxiv.org/pdf/1807.11205.pdf and https://arxiv.org/pdf/1812.01187.pdf
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in network.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.LayerNorm) or isinstance(v, nn.GroupNorm):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    print(f"Using Adam with weight decay: {args.weight_decay}")
    optim = torch.optim.AdamW(pg0, lr=train_parameters['lr'])

    optim.add_param_group({'params': pg1, 'weight_decay': args.weight_decay})  # add pg1 with weight_decay
    optim.add_param_group({'params': pg2})  # add pg2 (biases)

    lrf = args.learning_rate_factor
    lf = lambda x: ((1 + math.cos(x * math.pi / train_parameters['num_epochs'])) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = LambdaLR(optim, lr_lambda=lf)

    min_loss = np.infty

    for epoch in range(train_parameters['num_epochs']):

        if args.distributed:
            sampler_train.set_epoch(epoch)

        network.train()

        tr_stats = iterate_dataset(network, train_loader, optimizer=optim, criterion=criterion,
                                   clip_grads=args.clip_grads, device=device, tempo_aug=args.augment)

        if args.distributed:
            torch.distributed.barrier()

        network.eval()

        val_stats = iterate_dataset(network, val_loader, optimizer=None, criterion=criterion, device=device)

        # wait for evaluation to finish on all nodes
        if args.distributed:
            torch.distributed.barrier()

        tr_stats = {k: torch.FloatTensor([v]).to(device) for k, v in tr_stats.items() if isinstance(v, float)}
        val_stats = {k: torch.FloatTensor([v]).to(device) for k, v in val_stats.items() if isinstance(v, float)}

        tr_stats = reduce_dict(tr_stats, average=True)
        val_stats = reduce_dict(val_stats, average=True)

        tr_loss = tr_stats['loss'].item()

        val_loss = val_stats['loss'].item()

        train_diff = tr_stats['frame_diffs_mean'].item()
        val_diff = val_stats['frame_diffs_mean'].item()
        if val_loss < min_loss:
            min_loss = val_loss
            color = '\033[92m'

            if is_main_process() and logging:
                print("Store best model...")
                torch.save(network.state_dict(), os.path.join(train_parameters['dump_path'], "best_model.pt"))
        else:
            color = '\033[91m'

            # store latest model
            if is_main_process() and logging:
                torch.save(network.state_dict(),
                           os.path.join(train_parameters['dump_path'], "latest_model.pt".format(epoch)))

        if is_main_process() and logging:

            log_writer.add_scalar('training/lr', optim.param_groups[0]['lr'], epoch)

            log_writer.add_scalar(f'training/frame_diff', train_diff, epoch)
            log_writer.add_scalar(f'validation/frame_diff', val_diff, epoch)

            # log losses
            for key in tr_stats:
                if "loss" in key:
                    log_writer.add_scalar(f'training/{key}', tr_stats[key].item(), epoch)
                    log_writer.add_scalar(f'validation/{key}', val_stats[key].item(), epoch)

            # log weights
            if is_main_process() and args.log_weights:
                for name, W in network.named_parameters():
                    log_writer.add_histogram(f'model_params/{name}', W, epoch)

        print(f"\n{color}Epoch {epoch} | Train Loss: {tr_loss:.6f} | Frame-Diff: {train_diff:.4f}\033[0m")
        print(f"{color}Epoch {epoch} |   Val Loss: {val_loss:.6f} | Frame-Diff: {val_diff:.4f}\033[0m")

        scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Script')
    parser.add_argument('--augment', help='activate data augmentation', default=False, action='store_true')
    parser.add_argument('--config', help='path to config.', type=str)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--dump_root', help='name for the stored network', type=str, default="params")
    parser.add_argument('--ir_path', help='path to room impulse responses', type=str, default=None, nargs='+')
    parser.add_argument('--load_audio', default=False, action='store_true', help="preload audio files for datapool")
    parser.add_argument('--log_weights', help='log weights', default=False, action='store_true')
    parser.add_argument('--log_root', help='path to log directory', type=str, default="runs")
    parser.add_argument('--no_log', help='do not log', default=False, action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', help='random seed.', type=int, default=4711)
    parser.add_argument('--tag', help='experiment tag.', type=str)
    parser.add_argument('--scale_width', help='sheet image scale factor.', type=int, default=416)
    parser.add_argument('--train_set', help='path to train dataset.')
    parser.add_argument('--train_split_file', help='split file to only train on a subset from the train dir',
                        default=None)
    parser.add_argument('--val_set', help='path to validation dataset.')
    parser.add_argument('--val_split_file', help='split file to only evaluate a subset from the validation dir',
                        default=None)

    # arguments for optimizer and scheduler
    parser.add_argument('--batch_size', help='batch size.', type=int, default=32)
    parser.add_argument('--clip_grads', help='gradient clipping value', type=float, default=0.1)
    parser.add_argument('--learning_rate', "--lr", help='learning rate.', type=float, default=5e-4)
    parser.add_argument('--learning_rate_factor', "--lrf", help='factor for multiplying the learning rate',
                        type=float, default=0.01)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--weight_decay', help='weight decay value.', type=float, default=0.0005)

    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    args.config = load_yaml(args.config)
    train(args)
