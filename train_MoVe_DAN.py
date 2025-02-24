#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import argparse
import json
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import datetime
import glob
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from libs.config.DAN_config import OPTION as opt
from libs.utils.Logger import TreeEvaluation as Evaluation, TimeRecord, LogTime, Tee, Loss_record
from libs.utils.Restore import get_save_dir, restore, save_model
from libs.models.DAN import *
from libs.dataset.MoVe import MoVeDataset
from libs.dataset.transform import TrainTransform, TestTransform
from torch.utils.data import DataLoader
from libs.utils.optimer import DAN_optimizer
import numpy as np
from libs.utils.loss import *

from libs.models.DMA.loss import build_criterion

SNAPSHOT_DIR = opt.SNAPSHOT_DIR


def setup_ddp():
    """Initialize DDP environment"""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank


def cleanup_ddp():
    """Cleanup DDP environment"""
    dist.destroy_process_group()


def get_arguments():
    parser = argparse.ArgumentParser(description='MoVe Training')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--total_episodes", type=int, default=100000, help="total number of episodes for training")
    parser.add_argument("--support_frames", type=int, default=5)
    parser.add_argument("--query_frames", type=int, default=5)
    parser.add_argument("--num_ways", type=int, default=2, help="number of ways (N) in N-way-K-shot setting")
    parser.add_argument("--num_shots", type=int, default=2, help="number of shots (K) in N-way-K-shot setting")
    parser.add_argument("--max_test_episodes", type=int, default=1000, help="maximum number of episodes for testing")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--save_interval", type=int, default=200, help="Save model every N episodes")
    parser.add_argument("--test_interval", type=int, default=5000, help="Run test every N episodes")
    parser.add_argument("--print_interval", type=int, default=10, help="Print progress every N episodes")
    parser.add_argument("--setting", type=str, default="default", help="default or challenging")
    parser.add_argument("--loss_type", type=str, default="default", help="cross_entropy or focal")
    parser.add_argument("--ce_loss_weight", type=float, default=0.25, help="weight for cross entropy loss")
    parser.add_argument("--iou_loss_weight", type=float, default=5.0, help="weight for iou loss")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    return parser.parse_args()

def train():
    args = get_arguments()
    
    # Setup DDP
    local_rank = setup_ddp()
    
    # Set random seed for reproducibility
    seed = 1234
    # Set random seed for all processes
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Additional settings for multi-GPU deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Ensure all processes have same random seed
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    # Create snapshot directory only on main process
    save_dir = os.path.join(args.snapshot_dir, f'group{args.group}')
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        logger = Tee(os.path.join(save_dir, 'train_log.txt'), 'a')
        # Initialize tensorboard writer
        writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
        print('Running parameters:\n')
        print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # Build model
    model = DAN.DAN()
    
    if local_rank == 0:
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Move model to GPU and wrap with DDP
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Setup optimizer and gradient scaler for mixed precision training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_episodes, eta_min=1e-6)
    scaler = GradScaler()

    # # Try to load latest checkpoint
    # start_episode = 0
    # best_loss = float('inf')
    # checkpoint_pattern = os.path.join(save_dir, 'DAN_MoVe_episode*.pth')
    # checkpoints = glob.glob(checkpoint_pattern)
    # if checkpoints:
    #     latest_checkpoint = max(checkpoints, key=os.path.getctime)
    #     if local_rank == 0:
    #         print(f"Resuming from checkpoint: {latest_checkpoint}")
    #     checkpoint = torch.load(latest_checkpoint, map_location=f'cuda:{local_rank}')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_episode = checkpoint['episode']
    #     best_loss = checkpoint['loss']
    #     if 'scaler_state_dict' in checkpoint:
    #         scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    #     # Adjust learning rate scheduler
    #     for _ in range(start_episode):
    #         scheduler.step()
    
        # Try to load latest checkpoint if resume is True
    start_episode = 0
    best_loss = float('inf')
    if args.resume:
        checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            if local_rank == 0:
                print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            best_loss = checkpoint['loss']
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Adjust learning rate scheduler
            for _ in range(start_episode):
                scheduler.step()

    # Setup datasets
    size = (241, 425)
    train_transform = TrainTransform(size)
    test_transform = TestTransform(size)
    
    train_dataset = MoVeDataset(
        data_path=opt.root_path,
        train=True,
        group=args.group,
        support_frames=args.support_frames,
        query_frames=args.query_frames,
        num_ways=args.num_ways,
        num_shots=args.num_shots,
        transforms=train_transform,
        setting=args.setting
    )
    
    # test_dataset = MoVeDataset(
    #     data_path=opt.root_path,
    #     train=False,
    #     support_frames=args.support_frames,
    #     query_frames=args.query_frames,
    #     num_ways=args.num_ways,
    #     num_shots=args.num_shots,
    #     transforms=test_transform
    # )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     sampler=test_sampler
    # )

    # Training loop
    if local_rank == 0:
        print(f'Start training from episode {start_episode + 1}')
        start_time = time.time()
    
    model.train()
    total_loss = 0
    moving_loss = 0  # 用于计算移动平均损失
    moving_ce_loss = 0
    moving_iou_loss = 0

    celoss = cross_entropy_loss
    # criterion = lambda pred, target, bootstrap=1: [celoss(pred, target, bootstrap), mask_iou_loss(pred, target)]
    criterion = build_criterion(args.loss_type)
    for episode, (query_frames, query_masks, support_frames, support_masks, _) in enumerate(train_loader, start=start_episode):
        if episode >= args.total_episodes:
            break
            
        try:
            # 转换数据格式
            B, T, C, H, W = query_frames.shape
            query_frames = query_frames.view(args.batch_size, args.num_ways, -1, C, H, W)
            query_masks = query_masks.view(args.batch_size, args.num_ways, -1, H, W)
            
            _, SF, _, _, _ = support_frames.shape
            F = args.support_frames
            K = args.num_shots
            support_frames = support_frames.view(args.batch_size, args.num_ways, K, F, C, H, W)
            support_masks = support_masks.view(args.batch_size, args.num_ways, K, F, H, W)
            
            # 移动数据到GPU
            query_frames = query_frames.cuda(local_rank, non_blocking=True)
            query_masks = query_masks.cuda(local_rank, non_blocking=True)
            support_frames = support_frames.cuda(local_rank, non_blocking=True)
            support_masks = support_masks.cuda(local_rank, non_blocking=True)
            support_masks = support_masks.unsqueeze(4)
            
            # Forward pass
            optimizer.zero_grad()
            total_loss = 0
            
            # Process one way at a time
            for n in range(args.num_ways):
                # Extract current way's data
                current_query = query_frames[:, n].contiguous() # [B, T, C, H, W]
                current_support = support_frames[:, n].reshape(args.batch_size, -1, C, H, W).contiguous()  # [B, K*F, C, H, W]
                current_support_mask = support_masks[:, n].reshape(args.batch_size, -1, 1, H, W).contiguous()  # [B, K*F, 1, H, W]
                current_query_mask = query_masks[:, n].contiguous()  # [B, T, H, W]
                
                # Forward pass for current way with mixed precision
                with autocast():
                    output = model(current_query, current_support, current_support_mask).sigmoid()

                    # few_ce_loss, few_iou_loss = criterion(output.squeeze(2), current_query_mask)
                    few_ce_loss, few_iou_loss = criterion(output.squeeze(2), current_query_mask)
                    total_loss = args.ce_loss_weight * few_ce_loss + args.iou_loss_weight * few_iou_loss
                    way_loss = total_loss / args.num_ways
                
                # Accumulate loss
                total_loss += way_loss
                
                # Backward pass with gradient scaling
                scaler.scale(way_loss).backward()
            
            # Update parameters with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update loss statistics
            moving_ce_loss = 0.9 * moving_ce_loss + 0.1 * few_ce_loss.item() if episode > 0 else few_ce_loss.item()
            moving_iou_loss = 0.9 * moving_iou_loss + 0.1 * few_iou_loss.item() if episode > 0 else few_iou_loss.item()
            moving_loss = 0.9 * moving_loss + 0.1 * total_loss.item() if episode > 0 else total_loss.item()
            
            # Log to tensorboard
            if local_rank == 0:
                writer.add_scalar('Loss/CE', moving_ce_loss, episode)
                writer.add_scalar('Loss/IoU', moving_iou_loss, episode)
                writer.add_scalar('Loss/Total', moving_loss, episode)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], episode)
            
            # Print progress
            if local_rank == 0 and (episode + 1) % args.print_interval == 0:
                avg_loss = total_loss / args.print_interval
                total_loss = 0
                
                time_spent = time.time() - start_time
                time_per_episode = time_spent / (episode + 1 - start_episode)
                remaining_episodes = args.total_episodes - episode - 1
                eta_seconds = remaining_episodes * time_per_episode
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(f'Episode [{episode+1}/{args.total_episodes}], '
                      f'CE Loss: {moving_ce_loss:.4f}, '
                      f'IoU Loss: {moving_iou_loss:.4f}, '
                      f'Total Loss: {moving_loss:.4f}, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}, '
                      f'ETA: {eta}')
            
            # # Save checkpoint
            # if local_rank == 0 and (episode + 1) % args.save_interval == 0:
            #     if moving_loss < best_loss:
            #         best_loss = moving_loss
            #         checkpoint_path = os.path.join(save_dir, f'DAN_MoVe_episode{episode+1}_loss{moving_loss:.4f}.pth')
            #         torch.save({
            #             'episode': episode + 1,
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'scaler_state_dict': scaler.state_dict(),
            #             'loss': best_loss,
            #             'args': args,
            #         }, checkpoint_path)
            #         print(f"Saved checkpoint at episode {episode+1}")
            # Save checkpoint
            if local_rank == 0 and (episode + 1) % args.save_interval == 0:
                # Save latest checkpoint
                latest_checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
                torch.save({
                    'episode': episode + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': moving_loss,
                    'args': args,
                }, latest_checkpoint_path)
                print(f"Saved latest checkpoint at episode {episode+1}")

                # Save best checkpoint if current loss is better
                if moving_loss < best_loss:
                    best_loss = moving_loss
                    best_checkpoint_path = os.path.join(save_dir, f'DAN_MoVe_episode{episode+1}_loss{moving_loss:.4f}.pth')
                    torch.save({
                        'episode': episode + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': best_loss,
                        'args': args,
                    }, best_checkpoint_path)
                    print(f"Saved best checkpoint at episode {episode+1} with loss {moving_loss:.4f}")
            
            # Synchronize processes
            dist.barrier()
            
        except Exception as e:
            if local_rank == 0:
                print(f"Detailed error traceback:")
                import traceback
                traceback.print_exc()
                print(f"Error in episode {episode}: {str(e)}")
                print(f"Query frames shape: {query_frames.shape}")
                print(f"Query masks shape: {query_masks.shape}")
                print(f"Support frames shape: {support_frames.shape}")
                print(f"Support masks shape: {support_masks.shape}")
            continue
    
    # Close tensorboard writer
    if local_rank == 0:
        writer.close()
    
    # Cleanup
    cleanup_ddp()

if __name__ == '__main__':
    train()