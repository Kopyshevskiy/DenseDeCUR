from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.distributed as dist
import torch.nn.functional as F
import diffdist
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable

# no warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.decur               import DeCUR
from src.models.densecl             import DenseCL
from src.models.densedecur          import DenseDeCUR
from src.dataio.kaist_dataset_list  import KAISTDatasetFromList
from src.dataio.loader              import build_kaist_transforms

# method
parser = argparse.ArgumentParser(description='Multimodal Self-Supervised Pretraining')
parser.add_argument('--dataset', type=str, choices=['KAIST'])  
parser.add_argument('--method',  type=str, choices=['DeCUR','DenseCL','DenseDeCUR'])   
parser.add_argument('--densecl_stream', type=str, default='rgb', choices=['rgb','thermal'])                 

# training hyperparameters
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--batch-size', default=128, type=int)

# optimizer hyperparameters
parser.add_argument('--learning-rate-weights', default=0.002,   type=float)
parser.add_argument('--learning-rate-biases',  default=0.00048, type=float)

#parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--lr',  default=0.2, type=float)
parser.add_argument('--cos', action='store_true', default=False)
parser.add_argument('--schedule', default=[120,160], nargs='*', type=int)

# (dense)decur hyperparameters
parser.add_argument('--lambd', default=0.0051, type=float)
parser.add_argument('--projector', default='8192-8192-8192', type=str)
parser.add_argument('--dim_common', type=int, default=6144)

# training settings
parser.add_argument('--resume', type=str, default='',help='resume path.')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path)
parser.add_argument('--print-freq', default=20, type=int)

# dataset
parser.add_argument('--data-root',  type=str, default='/scratch/project/eu-25-19/kaist-cvpr15/images')  
parser.add_argument('--list-train', type=str, default='/mnt/proj3/eu-25-19/davide_secco/ADL-Project/Kaist_txt_lists/Training_split_25_forSSL.txt')

# dist training
parser.add_argument("--dist_url", default="env://", type=str)
parser.add_argument("--world_size", default=-1, type=int, help='set automatically')
parser.add_argument("--rank", default=0, type=int, help='set automatically')
parser.add_argument('--seed', type=int, default=42)

# new arguments (to play around with)
parser.add_argument('--baseline', action='store_true', default=False) 
parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd','lars'])



def init_distributed_mode(args):
    
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # scegli backend in base all'hardware
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        args.gpu_to_work_on = args.rank % gpu_count
        torch.cuda.set_device(args.gpu_to_work_on)
    else:
        args.gpu_to_work_on = None  # fallback CPU-only

    return   



def fix_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def main():
    global args

    args = parser.parse_args()

    init_distributed_mode(args)
    
    fix_random_seeds(args.seed)
    
    main_worker(gpu=None,args=args)



def main_worker(gpu, args):

    # create tb_writer
    if args.rank==0 and not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir,exist_ok=True)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoint_dir,'log'))
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)


    # select model
    if args.method == 'DeCUR':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = DeCUR(args).to(device)
    elif args.method == 'DenseCL':        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = DenseCL(pretrained=True).to(device)
    elif args.method == 'DenseDeCUR':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = DenseDeCUR(args).to(device)


    param_weights = []
    param_biases  = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    

    # select training modality
    if torch.cuda.is_available():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_to_work_on])
    else:
        model = model.cpu()
        model = torch.nn.parallel.DistributedDataParallel(model)


    # select optmizer
    if args.method == 'DeCUR':
        # args.lr = 0
        # optimizer = LARS(parameters, lr=args.lr, weight_decay=1e-4, weight_decay_filter=True, lars_adaptation_filter=True)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.method == 'DenseCL':
        args.lr = 0.015 # if batch size = 128
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.method == 'DenseDeCUR':
        if args.optimizer == 'lars':
            optimizer = LARS(parameters, lr=0, weight_decay=1e-4, weight_decay_filter=True, lars_adaptation_filter=True)
        elif args.optimizer == 'sgd':
            args.lr = 0.015
            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)


    # automatically resume from checkpoint if it exists
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0    
    

    # dataset
    rgb_t, th_t = build_kaist_transforms(img_size=224)

    if args.data_root is not None and args.list_train is not None:
        train_dataset = KAISTDatasetFromList(     
            data_root=args.data_root,
            list_file=args.list_train,
            rgb_transform=rgb_t,
            th_transform=th_t)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        num_workers=args.workers, 
        pin_memory=args.is_slurm_job, 
        sampler=train_sampler, 
        drop_last=True)


    print(f"[INFO] Training method: {args.method}"
            + (f", stream: {args.densecl_stream}" if args.method == 'DenseCL' else "")
            + (", baseline" if (args.method == 'DenseDeCUR' and args.baseline)
                else (", not baseline" if args.method == 'DenseDeCUR' else "")
            )
        )
    

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    stats = {}
    loss = None
    for epoch in range(start_epoch, args.epochs):
        
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)
        
        for step, (y1, y2) in enumerate(train_loader, start=epoch * len(train_loader)):
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            y1_1 = y1[0].to(device, non_blocking=True)
            y1_2 = y1[1].to(device, non_blocking=True)
            y2_1 = y2[0].to(device, non_blocking=True)
            y2_2 = y2[1].to(device, non_blocking=True)
                        
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                
                if args.method=='DeCUR': 
                    loss1, loss2, loss12, on_diag12_c = model.forward(y1_1, y1_2, y2_1, y2_2)
                    loss = (loss1 + loss2 + loss12) / 3

                elif args.method=='DenseCL': 
                    if args.densecl_stream == 'rgb':
                        im_q, im_k = y1_1, y1_2  # rgb
                    elif args.densecl_stream == 'thermal':
                        im_q, im_k = y2_1, y2_2  # thermal
                    loss_global, loss_dense, _ = model.forward(im_q, im_k) 
                    loss = loss_global + loss_dense 


                if args.method=='DenseDeCUR':
                    loss1, loss2, loss12, on_diag12_c = model.forward(y1_1, y1_2, y2_1, y2_2)
                    loss = (loss1 + loss2 + loss12) / 3


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            
            if step % args.print_freq == 0:
                if args.rank == 0:

                    if args.method=='DeCUR':
                        stats = dict(epoch=epoch, step=step, loss=loss.item(), loss1=loss1.item(), loss2=loss2.item(), loss12=loss12.item(), on_diag12_c=on_diag12_c.item())
                    elif args.method=='DenseCL':
                        stats = dict(epoch=epoch, step=step, loss=loss.item(), loss_global=loss_global.item(), loss_dense=loss_dense.item())
                    elif args.method=='DenseDeCUR':
                        stats = dict(epoch=epoch, step=step, loss=loss.item(), loss1=loss1.item(), loss2=loss2.item(), loss12=loss12.item(), on_diag12_c=on_diag12_c.item())

                    if step == 0 and epoch == 0 and args.rank == 0:
                        header = " ".join(f"{k:<12}" for k in stats.keys())
                        print(header, flush=True)
                        print(header, file=stats_file, flush=True)

                    line = " ".join(f"{v:<12.6f}" if isinstance(v, float) else f"{v:<12}" for v in stats.values())
                    print(line, flush=True)
                    print(line, file=stats_file, flush=True)

                    
            
        
        # save checkpoint
        if args.method=='DeCUR':
            if args.rank == 0 and (epoch % 100 == 0 or epoch == args.epochs - 1):
                state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / 'decur_checkpoint_{:04d}.pth'.format(epoch))
                tb_writer.add_scalars('training log',stats,epoch)
        elif args.method=='DenseCL' and args.densecl_stream == 'rgb':
            if args.rank == 0 and (epoch % 100 == 0 or epoch == args.epochs - 1):
                state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / 'densecl_rgb_checkpoint_{:04d}.pth'.format(epoch))
                tb_writer.add_scalars('training log',stats,epoch)
        elif args.method=='DenseCL' and args.densecl_stream == 'thermal':
            if args.rank == 0 and (epoch % 100 == 0 or epoch == args.epochs - 1):
                state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / 'densecl_thermal_checkpoint_{:04d}.pth'.format(epoch))
                tb_writer.add_scalars('training log',stats,epoch)
        elif args.method=='DenseDeCUR':
            if args.rank == 0 and (epoch % 100 == 0 or epoch == args.epochs - 1):

                tag  = "baseline" if args.baseline else "notbaseline"
                name = f"densedecur_{tag}_{args.optimizer}_ep{epoch:04d}.pth"

                state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / name)
                tb_writer.add_scalars('training log',stats,epoch)
 
            

def adjust_learning_rate(args, optimizer, epoch):

    is_lars = (optimizer.__class__.__name__ == 'LARS')
    lr = args.lr
    w = 1.0

    if is_lars:
        if args.cos:
            w *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        else:
            for milestone in args.schedule:
                w *= 0.1 if epoch >= milestone else 1.
        optimizer.param_groups[0]['lr'] = w * args.learning_rate_weights
        optimizer.param_groups[1]['lr'] = w * args.learning_rate_biases 
    else:
        warmup_epochs   = 10
        warmup_start_lr = 0.0

        if epoch < warmup_epochs: # linear warm-up
            lr_t = warmup_start_lr + (lr - warmup_start_lr) * (epoch + 1) / warmup_epochs
        else:
            if args.cos:
                t = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
                lr_t = lr * (0.5 * (1. + math.cos(math.pi * t)))
            else:
                for milestone in args.schedule:
                    w *= 0.1 if epoch >= milestone else 1.0
                lr_t = lr * w
            
        for g in optimizer.param_groups:
                g['lr'] = lr_t



class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



if __name__ == '__main__':
    main()
