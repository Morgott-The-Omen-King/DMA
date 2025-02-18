import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def set_random_seed(seed: int = 1234, deterministic: bool = True) -> None:
    """
    设置随机种子以确保实验的可重复性，特别是在多GPU训练环境中。
    
    Args:
        seed (int): 随机种子值，默认为1234
        deterministic (bool): 是否使用确定性算法，默认为True
    """
    # 设置 Python 的随机种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    # 设置 CUDA 的随机种子
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # 设置 CUDNN 后端的确定性
        cudnn.deterministic = True
        # 禁用 CUDNN 的基准测试模式
        cudnn.benchmark = False
        
    # 如果在分布式训练环境中，确保所有进程使用相同的随机种子
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
