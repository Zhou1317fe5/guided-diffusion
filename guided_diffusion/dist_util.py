"""
分布式训练的辅助函数。
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# 根据您的集群布局更改此设置。
# 给定 rank 的 GPU 是 (rank % GPUS_PER_NODE)。
GPUS_PER_NODE = 8

# 设置重试次数
SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    设置分布式进程组。
    """
    # 如果分布式进程组已经初始化，则直接返回
    if dist.is_initialized():
        return
    # 设置当前进程可见的 CUDA 设备
    # 每个进程根据其在 MPI.COMM_WORLD 中的 rank 分配一个 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD
    # 如果 CUDA 可用，则使用 'nccl' 后端，否则使用 'gloo'
    # 'nccl' 是 NVIDIA 的集合通信库，专为 GPU 优化
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        # 如果使用 'gloo' 后端，主机名设为 'localhost'
        hostname = "localhost"
    else:
        # 否则，获取完全限定域名对应的主机名
        hostname = socket.gethostbyname(socket.getfqdn())
    # 从 rank 0 进程广播主机名到所有其他进程，并设置 MASTER_ADDR 环境变量
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    # 设置当前进程的 RANK 环境变量
    os.environ["RANK"] = str(comm.rank)
    # 设置 WORLD_SIZE 环境变量，即总进程数
    os.environ["WORLD_SIZE"] = str(comm.size)

    # 从 rank 0 进程广播一个空闲端口号到所有其他进程
    port = comm.bcast(_find_free_port(), root=0)
    # 设置 MASTER_PORT 环境变量
    os.environ["MASTER_PORT"] = str(port)
    # 使用 "env://" 初始化方法初始化进程组，该方法会从环境变量中读取配置
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    获取用于 torch.distributed 的设备。
    """
    # 如果 CUDA 可用，返回 'cuda' 设备
    if th.cuda.is_available():
        return th.device(f"cuda")
    # 否则，返回 'cpu' 设备
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    加载 PyTorch 文件，避免在 MPI 各个 rank 之间产生冗余的读取操作。
    """
    chunk_size = 2 ** 20  # MPI 的大小限制相对较小，设置块大小为 1MB
    if MPI.COMM_WORLD.Get_rank() == 0:
        # rank 0 进程负责读取文件
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        # 计算数据块的数量
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        # 将数据块数量广播给所有其他进程
        MPI.COMM_WORLD.bcast(num_chunks)
        # 逐块广播数据
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        # 其他进程负责接收数据
        # 接收数据块数量
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        # 循环接收所有数据块
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    # 使用 io.BytesIO 将字节数据转换为文件类对象，然后用 torch.load 加载
    return th.load(io.BytesIO(data), **kwargs)



def sync_params(params):
    """
    将一系列 Tensors 从 rank 0 同步到所有 rank。
    """
    for p in params:
        with th.no_grad():
            # 使用 dist.broadcast 将张量 p 从 rank 0 广播到所有其他进程
            dist.broadcast(p, 0)


def _find_free_port():
    """
    查找一个空闲的端口。
    """
    try:
        # 创建一个套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 绑定到地址 "" 和端口 0，端口 0 表示由操作系统自动选择一个可用端口
        s.bind(("", 0))
        # 设置套接字选项，允许地址重用
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 返回获取到的端口号
        return s.getsockname()[1]
    finally:
        # 确保套接字被关闭
        s.close()
