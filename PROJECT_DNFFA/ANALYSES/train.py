import os
from os.path import exists
import sys
import json
import time
import argparse
import numpy as np
import wandb
from tqdm import tqdm
from IPython.core.debugger import set_trace

from fastprogress.fastprogress import progress_bar

from PROJECT_DNFFA.HELPERS import paths
from PROJECT_DNFFA.ANALYSES import losses

import ffcv.fields.decoders as Decoders
from ffcv.reader import Reader
from ffcv.fields import IntField, RGBImageField
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, RandomResizedCrop
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.fields import IntField, RGBImageField
from typing import List, Tuple

import torch

def arg_helper():
    
    parser = argparse.ArgumentParser(description='Evaluate model features on ImageNet')

    # high level hyperparams
    parser.add_argument('--device', default='cuda:0', 
                        type=str, metavar='N', help='device for training')
    
    parser.add_argument('--num-workers', default=12, 
                        type=int, metavar='N', help='number of dataloader workers')
    
    parser.add_argument('--log-freq', default=10, 
                        type=int, metavar='N', help='logging frequency (in batches)')
    
    # duration of training and batch size
    parser.add_argument('--train-epochs', default=10, 
                        type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--batch-size', default=512, 
                        type=int, metavar='N', help='mini-batch size')
    
    # learning rate specs
    parser.add_argument('--max-lr', default=0.05, 
                        type=float, metavar='LR', help='one-cycle max learning rate')
    parser.add_argument('--initial-lr', default=0.001, 
                        type=float, metavar='LR', help='one-cycle initial learning rate')
    parser.add_argument('--pct_start', default=0.3, 
                        type=float, metavar='LR', help='proportion epochs spent increasing lr')

    # loss function specs
    parser.add_argument('--sparse-pos', default=True, 
                        type=bool, metavar='W', help='enable sparse positive clf?')

    parser.add_argument('--l1-pos-lambda', default=0.0005, 
                        type=float, metavar='L', help='if sparse-pos is enabled, l1 lambda applied to positive weights')

    parser.add_argument('--l1-neg-lambda', default=0.001, 
                        type=float, metavar='L', help='if sparse-pos is enabled, l1 lambda applied to negative weights')

    args = parser.parse_args("")
    
    return args

def main_training_loop(model, train_loader, val_loader, args):
    
    description = f'mdl-{model.model_str}_from-{args.readout_from}_mlr-{args.max_lr}_ilr-{args.initial_lr}_eps-{args.train_epochs}_sparse-pos-{args.sparse_pos}'
    
    if args.sparse_pos is True:
        description = f'{description}_l1p-{args.l1_pos_lambda}_l1n-{args.l1_neg_lambda}'
        
    print(description)
    
    checkpoint_dir = f'{paths.training_checkpoint_dir()}/{description}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ################
    
    wandb.init(project=args.wandb_repo)
    wandb.config.update(args)
    wandb.run.name = description
    wandb.run.save()
    
    ################
    
    criterion = losses.get_loss_fn(args)
    
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.max_lr, 
                                momentum=0.9)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    steps_per_epoch=len(train_loader), 
                                                    max_lr=args.max_lr, 
                                                    epochs=args.train_epochs,
                                                    div_factor=args.max_lr / args.initial_lr)
    
    #################
    
    stats_file = open(f'{checkpoint_dir}/stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)
    print(args, file=stats_file)
    
    # automatically resume from checkpoint if it exists
    if exists(f'{checkpoint_dir}/checkpoint.pth'):
        ckpt = torch.load(f'{checkpoint_dir}/checkpoint.pth',
                          map_location=args.device)
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)
        
    ##################
    
    model.to(args.device)
    
    print('training readout...')

    start_time = time.time()
    
    step = 0
    for epoch in progress_bar(range(start_epoch, args.train_epochs)):

        model.eval()
        print(len(train_loader))
        for images, target in train_loader:

            output = model(images.to(args.device, non_blocking=True))
            
            if args.sparse_pos is True:
                readout_weights = model.readout.weight
                
                loss, clf_loss, l1_pos_loss, l1_neg_loss = criterion.compute_loss(readout_weights, 
                                                                             output, 
                                                                             target.to(args.device, non_blocking=True))
            else:
                loss = criterion(output, target.to(args.device, non_blocking=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % args.log_freq == 0:

                pg = optimizer.param_groups
                current_lr = pg[0]['lr']
                acc1, acc5 = accuracy(output, target.to(args.device, non_blocking=True), topk=(1, 5))
                print('acc1',acc1,'acc5',acc5)
                
                if args.sparse_pos is True:
                    stats = dict(epoch=epoch, step=step,
                                 current_lr=current_lr, 
                                 loss=loss.item(), clf_loss=clf_loss.item(),
                                 l1_pos_loss=l1_pos_loss.item(), l1_neg_loss=l1_neg_loss.item(),
                                 min_weight = torch.min(readout_weights).item(),
                                 prop_neg_weights=torch.mean((readout_weights < 0).float()).item(),
                                 acc1=acc1.item(), acc5=acc5.item(),
                                 time=int(time.time() - start_time))
                else:
                    stats = dict(epoch=epoch, step=step,
                                 current_lr=current_lr, 
                                 loss=loss.item(),
                                 acc1=acc1.item(), acc5=acc5.item(),
                                 time=int(time.time() - start_time))
                
                
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                
                wandb.log(stats)
                wandb.watch(model)
                step += 1

        print(f'evaluating for epoch {epoch}...')

        # evaluate
        top1 = AverageMeter('Acc@1')
        top5 = AverageMeter('Acc@5')
        with torch.no_grad():
            for images, target in progress_bar(val_loader):
                output = model(images.to(args.device, non_blocking=True))
                acc1, acc5 = accuracy(output, target.to(args.device, non_blocking=True), topk=(1, 5))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

        best_acc.top1 = max(best_acc.top1, top1.avg)
        best_acc.top5 = max(best_acc.top5, top5.avg)
        stats = dict(epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

        state = dict(
            epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        torch.save(state, f'{checkpoint_dir}/checkpoint.pth')

    stats_file.close()
        
    return model


def get_ffcv_dataloaders(args):

    # ffcv hyperparameters
    device = args.device
    batches_ahead = 2
    distributed = 0
    in_memory = 1
    precision = np.float32
    
    N_IMAGENET_CLASSES = 1000
    N_INPUT_FEATURES = 4096
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224/256
    
    # get decoders
    train_decoder = Decoders.RandomResizedCropRGBImageDecoder(output_size=(256,256), 
                                                              scale=(0.08, 1.0),
                                                              ratio=(0.75, 1.3333333333333333))

    val_decoder = Decoders.CenterCropRGBImageDecoder((224,224),
                                      ratio = DEFAULT_CROP_RATIO)
    
    # get paths
    train_data_path = paths.ffcv_imagenet1k_trainset()
    val_data_path = paths.ffcv_imagenet1k_valset()
    
    # make pipelines
    train_image_pipeline: List[Operation] = [
        train_decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, precision)
    ]

    val_image_pipeline: List[Operation] = [
        val_decoder,
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, precision)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]    

    # make dataloaders
    train_loader = Loader(train_data_path,
                batch_size=args.batch_size,
                batches_ahead=batches_ahead,
                num_workers=args.num_workers,
                order=OrderOption.RANDOM,
                os_cache=in_memory==1,
                drop_last=False,
                pipelines={
                    'image': train_image_pipeline,
                    'label': label_pipeline
                },
                custom_fields={
                    'image': RGBImageField,
                    'label': IntField,
                },
                distributed=distributed)

    val_loader = Loader(val_data_path,
                batch_size=args.batch_size,
                batches_ahead=batches_ahead,
                num_workers=args.num_workers,
                order=OrderOption.SEQUENTIAL,
                os_cache=in_memory==1,
                drop_last=False,
                pipelines={
                    'image': val_image_pipeline,
                    'label': label_pipeline
                },
                custom_fields={
                    'image': RGBImageField,
                    'label': IntField,
                },
                distributed=distributed)
    
    return train_loader, val_loader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
