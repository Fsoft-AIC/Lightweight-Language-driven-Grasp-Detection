import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from torch.nn.utils.rnn import pad_sequence

from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
from inference.models.PromptGD.prompt_gd import PromptGDModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--clip-version', type=str, default='ViT-B/32',
                        help='Version of CLIP model')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument("--use-mask", type=int, default=1,
                        help='Use part mask for training (1/0)')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr-step-size', type=int, default=10,
                        help='learning rate decay')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/ld_grasp',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--seen', type=int, default=1,
                        help='Flag for using seen classes, only work for Grasp-Anything dataset') 
    parser.add_argument('--alpha-vision-ckpt-pth', type=str, default=None,
                        help='Path to the ckpt file of the alpha vision model')
    parser.add_argument('--robot_checkpoint', type=str, default=None,
                        help='Path to the ckpt file of the robot model')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, iou_threshold):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor, text_embed in val_data:
            
            alpha_mask = x[:, -1:, :, :]
            
            x = x[:, :-1, :, :]
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            text_embed = text_embed.to(device)
            alpha_mask = alpha_mask.to(device)
            
            lossd = net.compute_loss(xc, yc, text_embed, alpha_mask)

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx[0], rot[0], zoom_factor[0]),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               threshold=iou_threshold
                                               )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
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

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        logging.info('Epoch: {}, Batch: {}/{}'.format(epoch, batch_idx, batches_per_epoch))
        for x, y, idx, _, _, text_embed in train_data:
            # print(text_embed.shape)
            # import sys
            # sys.exit(0)
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            
            # Split x into rgb and mask, split last channel as alpha mask, x has shape (N, C+1, 224, 224)
            
            alpha_mask = x[:, -1:, :, :]
            
            x = x[:, :-1, :, :]

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            text_embed = text_embed.to(device)
            alpha_mask = alpha_mask.to(device)
            
            lossd = net.compute_loss(xc, yc, text_embed, alpha_mask)

            loss = lossd['loss']

            if batch_idx % 20 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results

def custom_collate_fn(batch):
    # Unzip the batch
    x, pos_cos_sin_width, idx, rot, zoom_factor, text_embed = zip(*batch)

    # Convert x and pos_cos_sin_width to tensors
    x = torch.stack(x)
    pos, cos, sin, width = zip(*pos_cos_sin_width)
    pos = torch.stack(pos)
    cos = torch.stack(cos)
    sin = torch.stack(sin)
    width = torch.stack(width)

    # Pad the text embeddings
    text_embed = [torch.tensor(embed) for embed in text_embed]
    padded_text_embed = pad_sequence(text_embed, batch_first=True, padding_value=0)

    return x, (pos, cos, sin, width), idx, rot, zoom_factor, padded_text_embed

def run():
    args = parse_args()

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get the compute device
    # device = get_device(args.force_cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      random_rotate=True,
                      random_zoom=True,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      include_mask = args.use_mask,
                      seen=args.seen)
    logging.info('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler,
        collate_fn=custom_collate_fn,
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    net = PromptGDModel(
        clip_model=args.clip_version,
        alpha=True,
        alpha_vision_ckpt_pth=args.alpha_vision_ckpt_pth,
        robot_checkpoint=args.robot_checkpoint
    )
    net = net.to(device)
    params = 0
    for p in net.parameters():
        params += p.numel()
    logging.info(f'Total Params: {params}')

    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, 0.5)

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        logging.info('lr={}'.format(lr))
        tb.add_scalar('lr', lr, epoch)
        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou:
            torch.save(net, os.path.join(save_folder, 'epoch_best_iou.pth'))
            best_iou = iou
            print(best_iou)


if __name__ == '__main__':
    run()
