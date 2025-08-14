import argparse
import datetime
import json
import logging
import os
import sys
import functools

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torch.optim import AdamW
from torchsummary import summary

from accelerate import Accelerator
from transformers import set_seed

from diffusion.resample import create_named_schedule_sampler
from diffusion.fp16_util import MixedPrecisionTrainer

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
from utils.model_util import create_diffusion


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='lgdm',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--train-ratio', type=str, default="train",
                        help='Data split: train, test_seen, test_unseen') 
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anywhere',
                        help='Specific for Grasp-Anywhere')
    
    args = parser.parse_args()
    return args

def validate(net, diffusion, schedule_sampler, device, val_data, iou_threshold, accelerator: Accelerator):
    """
    Run validation in distributed setting using HuggingFace Accelerate.
    """
    net.eval()
    sample_fn = diffusion.p_sample_loop

    local_results = {
        'correct': 0,
        'failed': 0,
        'loss': 0.0,
        'losses': {}
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor, prompt, query in val_data:
            img = x.to(device)
            yc = [yy.to(device) for yy in y]
            pos_gt = yc[0]

            alpha = 0.4
            idx = torch.ones(img.shape[0]).to(device)

            sample = sample_fn(
                net,
                pos_gt.shape,
                pos_gt,
                img,
                query,
                alpha,
                idx,
            )

            pos_output = sample
            cos_output, sin_output, width_output = net.cos_output_str, net.sin_output_str, net.width_output_str

            lossd = net.compute_loss(yc, pos_output, cos_output, sin_output, width_output)
            loss = lossd['loss']

            local_results['loss'] += loss.item()

            for ln, l in lossd['losses'].items():
                if ln not in local_results['losses']:
                    local_results['losses'][ln] = 0.0
                local_results['losses'][ln] += l.item()

            q_out, ang_out, w_out = post_process_output(
                lossd['pred']['pos'], lossd['pred']['cos'],
                lossd['pred']['sin'], lossd['pred']['width']
            )

            s = evaluation.calculate_iou_match(
                q_out,
                ang_out,
                val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                no_grasps=1,
                grasp_width=w_out,
                threshold=iou_threshold
            )

            if s:
                local_results['correct'] += 1
            else:
                local_results['failed'] += 1

    # =========================
    # Synchronize across GPUs
    # =========================
    local_tensor = torch.tensor([
        local_results['correct'],
        local_results['failed'],
        local_results['loss']
    ], dtype=torch.float32, device=device)

    gathered = accelerator.gather_for_metrics(local_tensor)
    total_correct = gathered[:, 0].sum().item()
    total_failed = gathered[:, 1].sum().item()
    total_loss = gathered[:, 2].mean().item()

    # Sync nested loss dicts
    total_losses = {}
    for key in local_results['losses']:
        loss_val = torch.tensor(local_results['losses'][key], dtype=torch.float32, device=device)
        gathered_loss = accelerator.gather_for_metrics(loss_val)
        total_losses[key] = gathered_loss.mean().item()

    final_results = {
        'correct': int(total_correct),
        'failed': int(total_failed),
        'loss': total_loss,
        'losses': total_losses
    }

    return final_results

# def validate(net, diffusion, schedule_sampler, device, val_data, iou_threshold):
#     """
#     Run validation.
#     :param net: Network
#     :param device: Torch device
#     :param val_data: Validation Dataset
#     :param iou_threshold: IoU threshold
#     :return: Successes, Failures and Losses
#     """
#     net.eval()

#     sample_fn = (
#             diffusion.p_sample_loop
#     )

#     results = {
#         'correct': 0,
#         'failed': 0,
#         'loss': 0,
#         'losses': {

#         }
#     }

#     ld = len(val_data)

#     with torch.no_grad():
#         for x, y, didx, rot, zoom_factor, prompt, query in val_data:
#             img = x.to(device)
#             yc = [yy.to(device) for yy in y]
#             pos_gt = yc[0]

#             alpha = 0.4
#             idx = torch.ones(img.shape[0]).to(device)

#             sample = sample_fn(
#                 net,
#                 pos_gt.shape,
#                 pos_gt,
#                 img,
#                 query,
#                 alpha,
#                 idx,
#             )

#             pos_output = sample
#             cos_output, sin_output, width_output = net.cos_output_str, net.sin_output_str, net.width_output_str

#             lossd = net.compute_loss(yc, pos_output, cos_output, sin_output, width_output)
#             loss = lossd['loss']

#             results['loss'] += loss.item() / ld
#             for ln, l in lossd['losses'].items():
#                 if ln not in results['losses']:
#                     results['losses'][ln] = 0
#                 results['losses'][ln] += l.item() / ld

#             q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
#                                                         lossd['pred']['sin'], lossd['pred']['width'])

#             s = evaluation.calculate_iou_match(q_out,
#                                                ang_out,
#                                                val_data.dataset.get_gtbb(didx, rot, zoom_factor),
#                                                no_grasps=1,
#                                                grasp_width=w_out,
#                                                threshold=iou_threshold
#                                                )

#             if s:
#                 results['correct'] += 1
#             else:
#                 results['failed'] += 1

#     return results


def train(epoch, net, diffusion, schedule_sampler, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    # Setup for DDPM
    sample_fn = (
            diffusion.p_sample_loop
    )

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for x, y, _, _, _, prompt, query in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            img = x.to(device)
            yc = [yy.to(device) for yy in y]
            pos_gt = yc[0]

            if epoch>0:
                alpha = 0.4
            else:
                alpha = 0.4*min(1,batch_idx/len(train_data))
            idx = torch.zeros(img.shape[0]).to(device)
            t, weights = schedule_sampler.sample(img.shape[0], device)

            # Calculate loss
            compute_losses = functools.partial(
                diffusion.training_losses,
                net,
                pos_gt,
                img,
                t,  # [bs](int) sampled timesteps
                query,
                alpha,
                idx,
            )
            losses = compute_losses()
            loss = (losses["loss"] * weights).mean()

            pos_output, cos_output, sin_output, width_output = net.pos_output_str, net.cos_output_str, net.sin_output_str, net.width_output_str

            # Backward loss
            # mp_trainer.backward(loss)
            # mp_trainer.optimize(optimizer)

            lossd = net.compute_loss(yc, pos_output, cos_output, sin_output, width_output)
            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.mean().item()))

            results['loss'] += loss
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.random_seed)

    # Logging + TensorBoard
    if accelerator.is_main_process:
        dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
        save_folder = os.path.join(args.logdir, f"{dt}_{args.description.replace(' ', '_')}")
        os.makedirs(save_folder, exist_ok=True)
        tb = tensorboardX.SummaryWriter(save_folder)
    else:
        save_folder = None
        tb = None

    # Dataset
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(
        args.dataset_path,
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=True,
        random_zoom=True,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb,
        split=args.split,
        add_file_path=args.add_file_path
    )

    indices = list(range(dataset.length))
    split = int(np.floor(args.train_ratio * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[:split], indices[split:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=val_sampler, num_workers=args.num_workers)

    # Network + optimizer + diffusion
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net = network(input_channels=input_channels, dropout=args.use_dropout, prob=args.dropout_prob, channel_size=args.channel_size)

    diffusion = create_diffusion()
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError(f"Unknown optimizer {args.optim}")

    # Prepare with accelerate
    net, optimizer, train_loader, val_loader = accelerator.prepare(net, optimizer, train_loader, val_loader)

    best_iou = 0.0

    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0.0
        for batch_idx, (x, y, _, _, _, prompt, query) in enumerate(train_loader):
            if batch_idx >= args.batches_per_epoch:
                break

            img = x.to(device)
            yc = [yy.to(device) for yy in y]
            pos_gt = yc[0]

            alpha = 0.4 if epoch > 0 else 0.4 * min(1, batch_idx / len(train_loader))
            idx = torch.zeros(img.shape[0]).to(device)
            t, weights = schedule_sampler.sample(img.shape[0], device)

            compute_losses = functools.partial(
                diffusion.training_losses,
                net,
                pos_gt,
                img,
                t,
                query,
                alpha,
                idx
            )
            losses = compute_losses()
            loss = (losses["loss"] * weights).mean()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_loss /= (batch_idx + 1)

        if accelerator.is_main_process:
            tb.add_scalar('loss/train_loss', epoch_loss, epoch)
            print(f"[Epoch {epoch}] Train Loss: {epoch_loss:.4f}")

        # Validation
        val_results = validate(net, diffusion, schedule_sampler, device, val_loader, args.iou_threshold, accelerator)

        iou = val_results['correct'] / (val_results['correct'] + val_results['failed'])

        if accelerator.is_main_process:
            tb.add_scalar('loss/val_loss', val_results['loss'], epoch)
            tb.add_scalar('loss/IOU', iou, epoch)
            print(f"[Epoch {epoch}] Val IOU: {iou:.4f}")

            if iou > best_iou or epoch == 0 or (epoch % 10 == 0):
                model_path = os.path.join(save_folder, f"epoch_{epoch:02d}_iou_{iou:.2f}.pt")
                accelerator.save(net.state_dict(), model_path)
                best_iou = iou


if __name__ == '__main__':
    run()
