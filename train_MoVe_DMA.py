import warnings
warnings.filterwarnings("ignore")

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
from torch.optim.lr_scheduler import LambdaLR

from libs.config.DAN_config import OPTION as opt
from libs.utils.Logger import TreeEvaluation as Tee, Loss_record
from libs.utils.misc import set_random_seed
from libs.models.DMA import *
from libs.dataset.MoVe import MoVeDataset
from libs.dataset.transform import TrainTransform
from torch.utils.data import DataLoader
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--total_episodes", type=int, default=100000, help="total number of episodes for training")
    parser.add_argument("--support_frames", type=int, default=5)
    parser.add_argument("--query_frames", type=int, default=5)
    parser.add_argument("--num_ways", type=int, default=2, help="number of ways (N) in N-way-K-shot setting")
    parser.add_argument("--num_shots", type=int, default=2, help="number of shots (K) in N-way-K-shot setting")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--save_interval", type=int, default=200, help="Save model every N episodes")
    parser.add_argument("--print_interval", type=int, default=10, help="Print progress every N episodes")
    parser.add_argument("--ce_loss_weight", type=float, default=1.0, help="Weight for cross entropy loss")
    parser.add_argument("--iou_loss_weight", type=float, default=1.0, help="Weight for IoU loss")
    parser.add_argument("--setting", type=str, default="default", help="default or challenging")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--warmup_episodes", type=int, default=50, help="Number of warmup episodes")
    parser.add_argument("--loss_type", type=str, default="default", help="default or mask2former")
    return parser.parse_args()

def get_warmup_cosine_schedule_with_warmup(optimizer, warmup_episodes, total_episodes):
    def lr_lambda(episode):
        if episode < warmup_episodes:
            return float(episode) / float(max(1, warmup_episodes))
        progress = float(episode - warmup_episodes) / float(max(1, total_episodes - warmup_episodes))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train():
    args = get_arguments()
    
    # Setup DDP
    local_rank = setup_ddp()
    
    # 使用新的函数设置随机种子
    set_random_seed(seed=1234, deterministic=True)
    
    # Create snapshot directory only on main process
    save_dir = os.path.join(args.snapshot_dir, f'group{args.group}')
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        # 使用Tee保存完整的训练日志
        log_file = os.path.join(save_dir, 'train_log.txt')
        
        # 保存完整的参数配置
        with open(log_file, 'a') as f:
            f.write('='*50 + '\n')
            f.write('Running parameters:\n') 
            f.write('='*50 + '\n')
            for arg in vars(args):
                f.write(f'{arg}: {getattr(args, arg)}\n')
            f.write('='*50 + '\n')
            f.write('\nFull configuration:\n')
            f.write(json.dumps(vars(args), indent=4, separators=(',', ':')) + '\n')
            f.write('='*50 + '\n\n')
        
        # Initialize tensorboard writer
        writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    # Build model
    model = DMA(n_way=args.num_ways, k_shot=args.num_shots)
    
    if local_rank == 0:
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Move model to GPU and wrap with DDP
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Setup optimizer and gradient scaler for mixed precision training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_warmup_cosine_schedule_with_warmup(optimizer, args.warmup_episodes, args.total_episodes)
    scaler = GradScaler()

    # Try to load latest checkpoint if resume is True
    start_episode = 0
    best_loss = float('inf')
    if args.resume:
        checkpoint_pattern = os.path.join(save_dir, 'DAN_MoVe_episode*.pth')
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            if local_rank == 0:
                print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=f'cuda:{local_rank}')
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
    train_dataset = MoVeDataset(
        train=True,
        group=args.group,
        support_frames=args.support_frames,
        query_frames=args.query_frames,
        num_ways=args.num_ways,
        num_shots=args.num_shots,
        transforms=train_transform,
        setting=args.setting
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    # Training loop
    if local_rank == 0:
        print(f'Start training from episode {start_episode + 1}')
        start_time = time.time()
    
    model.train()
    total_loss = 0
    moving_loss = 0  # 用于计算移动平均损失
    moving_ce_loss = 0
    moving_iou_loss = 0

    criterion = build_criterion(args.loss_type)
    
    # 创建loss记录器
    loss_recorder = Loss_record()
    
    for episode, (query_frames, query_masks, support_frames, support_masks, _) in enumerate(train_loader, start=start_episode):
        if episode >= args.total_episodes:
            break
            
        try:
            # 转换数据格式
            _, _, C, H, W = query_frames.shape
            query_frames = query_frames.view(args.batch_size, args.num_ways, -1, C, H, W)
            query_masks = query_masks.view(args.batch_size, args.num_ways, -1, H, W)
            
            F = args.support_frames
            K = args.num_shots
            support_frames = support_frames.view(args.batch_size, args.num_ways, K, F, C, H, W)
            support_masks = support_masks.view(args.batch_size, args.num_ways, K, F, H, W)
            
            # 移动数据到GPU
            query_frames = query_frames.cuda(local_rank, non_blocking=True)
            query_masks = query_masks.cuda(local_rank, non_blocking=True)
            support_frames = support_frames.cuda(local_rank, non_blocking=True)
            support_masks = support_masks.cuda(local_rank, non_blocking=True)
            support_masks = support_masks
            
            # Forward pass
            optimizer.zero_grad()
            total_loss = 0
            
            # Forward pass with mixed precision
            with autocast():
                output = model(query_frames[:, 0], support_frames, support_masks)
                few_ce_loss, few_iou_loss = criterion(output.flatten(1, 2), query_masks.flatten(1, 2))
                total_loss = args.ce_loss_weight * few_ce_loss + args.iou_loss_weight * few_iou_loss
            
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            
            # Update parameters with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update loss statistics
            moving_ce_loss = 0.9 * moving_ce_loss + 0.1 * few_ce_loss.item() if episode > 0 else few_ce_loss.item()
            moving_iou_loss = 0.9 * moving_iou_loss + 0.1 * few_iou_loss.item() if episode > 0 else few_iou_loss.item()
            moving_loss = 0.9 * moving_loss + 0.1 * total_loss.item() if episode > 0 else total_loss.item()
            
            # 更新loss记录器
            loss_recorder.total.update(total_loss.item())
            loss_recorder.part1.update(few_ce_loss.item())
            loss_recorder.part2.update(few_iou_loss.item())
            
            # Log to tensorboard
            if local_rank == 0:
                writer.add_scalar('Loss/CE', moving_ce_loss, episode)
                writer.add_scalar('Loss/IoU', moving_iou_loss, episode)
                writer.add_scalar('Loss/Total', moving_loss, episode)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], episode)
            
            # Print progress and write to log file
            if local_rank == 0 and (episode + 1) % args.print_interval == 0:
                total_loss = 0
                
                time_spent = time.time() - start_time
                time_per_episode = time_spent / (episode + 1 - start_episode)
                remaining_episodes = args.total_episodes - episode - 1
                eta_seconds = remaining_episodes * time_per_episode
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                log_message = (f'Episode [{episode+1}/{args.total_episodes}], '
                             f'CE Loss: {moving_ce_loss:.4f}, '
                             f'IoU Loss: {moving_iou_loss:.4f}, '
                             f'Total Loss: {moving_loss:.4f}, '
                             f'LR: {scheduler.get_last_lr()[0]:.6f}, '
                             f'ETA: {eta}')
                print(log_message)
            
            # Save checkpoint
            if local_rank == 0 and (episode + 1) % args.save_interval == 0:
                if moving_loss < best_loss:
                    best_loss = moving_loss
                    checkpoint_path = os.path.join(save_dir, f'DAN_MoVe_episode{episode+1}_loss{moving_loss:.4f}.pth')
                    torch.save({
                        'episode': episode + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': best_loss,
                        'args': args,
                    }, checkpoint_path)
                    print(f"Saved checkpoint at episode {episode+1}")
            
            # Synchronize processes
            dist.barrier()
            
        except Exception as e:
            if local_rank == 0:
                print(f"Error in episode {episode}: {str(e)}")
                print(f"错误发生在第 {episode} 轮训练")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误堆栈信息:")
                import traceback
                print(traceback.format_exc())
                print("\n数据形状信息:")
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
