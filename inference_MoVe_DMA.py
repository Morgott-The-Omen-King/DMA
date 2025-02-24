#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import random
import json
import pickle
import torch.multiprocessing as mp
import datetime
import time
from pathlib import Path
from pycocotools import mask as mask_utils
import torch.nn.functional as F
from libs.dataset.transform import TestTransform
from libs.dataset.MoVe import MoVeDataset
from libs.models.DMA import DMA
from libs.config.DAN_config import OPTION as opt


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
    parser = argparse.ArgumentParser(description='Test DAN-MoVe with DDP')
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--support_frames", type=int, default=5, help="number of frames per shot")
    parser.add_argument("--query_frames", type=int, default=5, help="number of query frames")
    parser.add_argument("--num_ways", type=int, default=1, help="number of ways (N) in N-way-K-shot setting")
    parser.add_argument("--num_shots", type=int, default=2, help="number of shots (K) in N-way-K-shot setting")
    parser.add_argument("--snapshot", type=str, required=True, help="path to the trained model")
    parser.add_argument("--num_episodes", type=int, default=1, help="number of test episodes")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="temp2", help="directory to save inference results")
    parser.add_argument("--setting", type=str, default="default", help="default or challenging")
    return parser.parse_args()

def test():
    args = get_arguments()
    
    # Setup DDP and get local rank
    local_rank = setup_ddp()
    
    # Set random seed for reproducibility
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Ensure all processes have same random seed
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Setup Model
    model = DMA(n_way=args.num_ways, k_shot=args.num_shots).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Load trained model
    if local_rank == 0:
        print(f"Loading trained model from {args.snapshot}")
    checkpoint = torch.load(args.snapshot, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup Dataloader
    size = (241, 425)  # Define target size for rescaling
    test_transform = TestTransform(size)
    
    test_dataset = MoVeDataset(
        data_path=opt.root_path,
        train=False,
        group=args.group,
        support_frames=args.support_frames,
        query_frames=args.query_frames,
        num_ways=args.num_ways,
        num_shots=args.num_shots,
        transforms=test_transform,
        setting=args.setting
    )
    
    # Setup distributed sampler
    sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # Ensure multiprocessing compatibility
    mp.set_start_method('spawn', force=True)
    
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one episode at a time
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler
    )
    
    # Create save directory
    save_dir = Path(args.save_dir) / f'{args.num_ways}-way-{args.num_shots}-shot' / f'group{args.group}'
    if local_rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        print("Start testing...")
        start_time = time.time()
    
    results = []
    with torch.no_grad():
        for i, (query_frames, query_masks, support_frames, support_masks, video_ids, categories) in enumerate(test_loader):
            if i >= args.num_episodes:
                break
                
            try:
                # Move data to GPU
                query_frames = query_frames.to(device)
                query_masks = query_masks.to(device)
                support_frames = support_frames.to(device)
                support_masks = support_masks.to(device)
                
                # Get dimensions
                B, NT, C, H, W = query_frames.shape
                N = args.num_ways
                K = args.num_shots
                sup_F = args.support_frames
                T = NT // N
                
                # Reshape tensors for N-way processing
                query_frames = query_frames.view(B, N, T, C, H, W)
                query_masks = query_masks.view(B, N, T, H, W)
                support_frames = support_frames.view(B, N, K, sup_F, C, H, W)
                support_masks = support_masks.view(B, N, K, sup_F, H, W)
                
                # Process in chunks to avoid OOM
                chunk_size = 50
                pred_maps = []
                
                for start_idx in range(0, T, chunk_size):
                    end_idx = min(start_idx + chunk_size, T)
                    chunk_query = query_frames[:, 0, start_idx:end_idx]
                    
                    pred_map, _ = model(chunk_query, support_frames, support_masks) # B, N_way, T, H, W
                    
                    # Resize pred_map to match expected size [B, N_way, T, 1, 241, 425] using interpolation
                    pred_map = F.interpolate(pred_map.view(-1, 1, *pred_map.shape[-2:]), size=(241, 425), mode='bilinear', align_corners=False)
                    pred_map = pred_map.view(B, N, -1, 1, 241, 425)  # Reshape back to [B, N_way, T, 1, 241, 425]
                    
                    # pred_map = torch.sigmoid(pred_map)
                    pred_maps.append(pred_map.sigmoid())
        
                # Stack predictions for all ways
                pred_maps = torch.cat(pred_maps, dim=2)  # torch.Size([1, 2, 60, 1, 241, 425])
                
                # 获取形状信息
                B, N, T, _, H, W = pred_maps.shape
                
                # 创建one-hot编码的mask
                one_hot_masks = torch.zeros(B, N+1, T, H, W, device=pred_maps.device)
                
                # 设置背景类(索引0)
                background_mask = (pred_maps.max(dim=1)[0] < 0.5).squeeze(2)  # B x T x H x W
                one_hot_masks[:,0] = background_mask
                
                # 设置前景类(索引1到N)
                for n in range(N):
                    one_hot_masks[:,n+1] = (pred_maps[:,n,:,0] > 0.5)
                
                # 确保每个位置只有一个类别为1
                one_hot_masks = one_hot_masks.float()
                max_vals, max_indices = torch.max(one_hot_masks, dim=1, keepdim=True)
                one_hot_masks = torch.zeros_like(one_hot_masks).scatter_(1, max_indices, 1.0)
                
                pred_maps = one_hot_masks
                
                # query_masks torch.Size([1, 2, 60, 241, 425])
                # 将query_masks转换为one-hot格式
                one_hot_query_masks = torch.zeros(B, N+1, T, H, W, device=query_masks.device)
                
                # 设置背景类(索引0)
                background_mask = (query_masks.max(dim=1)[0] < 0.5)  # B x T x H x W
                one_hot_query_masks[:,0] = background_mask
                
                # 设置前景类(索引1到N)
                for n in range(N):
                    one_hot_query_masks[:,n+1] = query_masks[:,n]
                
                # 确保每个位置只有一个类别为1
                max_vals, max_indices = torch.max(one_hot_query_masks, dim=1, keepdim=True)
                one_hot_query_masks = torch.zeros_like(one_hot_query_masks).scatter_(1, max_indices, 1.0)
                
                query_masks = one_hot_query_masks
                
                # 将预测结果和真实标签转换为二值掩码
                binary_masks = (pred_maps.cpu().numpy()).astype(np.uint8)
                gt_masks = (query_masks.cpu().numpy()).astype(np.uint8)
                
                # 将二值掩码转换为RLE格式
                rle_masks = []
                rle_gt_masks = []
                for b in range(binary_masks.shape[0]):
                    rle_per_batch = []
                    rle_gt_per_batch = []
                    for n in range(binary_masks.shape[1]):
                        rle_per_way = []
                        rle_gt_per_way = []
                        for t in range(binary_masks.shape[2]):
                            rle = mask_utils.encode(np.asfortranarray(binary_masks[b,n,t]))
                            rle_gt = mask_utils.encode(np.asfortranarray(gt_masks[b,n,t]))
                            rle_per_way.append(rle)
                            rle_gt_per_way.append(rle_gt)
                        rle_per_batch.append(rle_per_way)
                        rle_gt_per_batch.append(rle_gt_per_way)
                    rle_masks.append(rle_per_batch)
                    rle_gt_masks.append(rle_gt_per_batch)
                
                # 保存结果
                episode_result = {
                    'predictions': rle_masks,
                    'query_masks': rle_gt_masks,
                    'video_ids': video_ids,
                    'categories': categories,
                    'class_list': test_dataset.test_categories
                }
                results.append(episode_result)
                
                # 打印进度和预计剩余时间
                if local_rank == 0 and (i + 1) % 10 == 0:
                    time_spent = time.time() - start_time
                    time_per_episode = time_spent / (i + 1)
                    remaining_episodes = args.num_episodes - i - 1
                    eta_seconds = remaining_episodes * time_per_episode
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                    print(f'Episode [{i+1}/{args.num_episodes}], '
                          f'Frames per query: {T}, '
                          f'ETA: {eta}')
                    
            except Exception as e:
                if local_rank == 0:
                    print(f"Error in episode {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue

    # 收集所有进程的结果
    world_size = dist.get_world_size()
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)
    
    # 保存所有进程的合并结果
    if local_rank == 0:
        results_file = save_dir / 'inference_results.pkl'
            
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nResults saved to {results_file}")
    
    # Cleanup
    cleanup_ddp()

if __name__ == '__main__':
    test()