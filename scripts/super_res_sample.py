"""
从超分辨率模型生成大批量样本。
需要一个来自 image_sample.py 的常规模型的样本批次作为输入。
"""

# 导入必要的库
import argparse  # 用于解析命令行参数
import os  # 用于与操作系统交互

import blobfile as bf  # 用于读写文件，特别是云存储中的文件
import numpy as np  # 用于数值计算
import torch as th  # PyTorch 深度学习框架
import torch.distributed as dist  # PyTorch 分布式计算库

# 从 guided_diffusion 库中导入工具函数和模块
from guided_diffusion import dist_util, logger  # 分布式工具和日志记录器
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,  # 超分辨率模型和扩散模型的默认参数
    sr_create_model_and_diffusion,  # 创建超分辨率模型和扩散过程的函数
    args_to_dict,  # 将 argparse 参数转换为字典
    add_dict_to_argparser,  # 将字典中的参数添加到 argparse 解析器
)


def main():
    """
    主函数，执行超分辨率采样过程。
    """
    # 创建并解析命令行参数
    args = create_argparser().parse_args()

    # 设置分布式环境
    dist_util.setup_dist()
    # 配置日志记录器
    logger.configure()

    # 记录日志：开始创建模型
    logger.log("creating model...")
    # 创建超分辨率模型和扩散过程
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    # 加载预训练的模型权重
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # 将模型移动到指定的计算设备（CPU 或 GPU）
    model.to(dist_util.dev())
    # 如果使用 FP16 精度，则转换模型
    if args.use_fp16:
        model.convert_to_fp16()
    # 将模型设置为评估模式
    model.eval()

    # 记录日志：开始加载数据
    logger.log("loading data...")
    # 为当前工作进程加载低分辨率图像数据
    data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)

    # 记录日志：开始生成样本
    logger.log("creating samples...")
    all_images = []  # 用于存储所有生成的图像
    # 循环生成样本，直到达到指定的数量
    while len(all_images) * args.batch_size < args.num_samples:
        # 从数据加载器中获取下一个批次的数据
        model_kwargs = next(data)
        # 将模型关键字参数移动到指定的计算设备
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        # 使用扩散模型的 p_sample_loop 方法生成样本
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.large_size, args.large_size),  # 样本形状
            clip_denoised=args.clip_denoised,  # 是否裁剪去噪后的结果
            model_kwargs=model_kwargs,  # 模型的附加参数，如低分辨率图像
        )
        # 将样本从 [-1, 1] 范围转换到 [0, 255] 范围
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # 调整张量维度顺序以匹配图像格式 (N, H, W, C)
        sample = sample.permute(0, 2, 3, 1)
        # 确保张量在内存中是连续的
        sample = sample.contiguous()

        # 准备一个列表来收集所有进程的样本
        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # 从所有进程中收集样本（注意：NCCL 后端不支持 all_gather）
        dist.all_gather(all_samples, sample)
        # 将收集到的样本添加到列表中
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        # 记录日志：已创建的样本数量
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # 将所有图像列表连接成一个 NumPy 数组
    arr = np.concatenate(all_images, axis=0)
    # 截取到所需的样本数量
    arr = arr[: args.num_samples]
    # 只有主进程（rank 0）保存结果
    if dist.get_rank() == 0:
        # 构建输出文件名
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        # 记录日志：保存文件路径
        logger.log(f"saving to {out_path}")
        # 将生成的样本数组保存为 .npz 文件
        np.savez(out_path, arr)

    # 等待所有进程完成
    dist.barrier()
    # 记录日志：采样完成
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    """
    为每个分布式工作进程加载数据。
    这是一个生成器函数，会持续不断地产生数据批次。

    :param base_samples: 包含低分辨率样本的 .npz 文件路径。
    :param batch_size: 每个批次的大小。
    :param class_cond: 是否有类别条件。
    """
    # 使用 blobfile 打开 .npz 文件
    with bf.BlobFile(base_samples, "rb") as f:
        # 加载 NumPy 存档
        obj = np.load(f)
        # 提取图像数组
        image_arr = obj["arr_0"]
        # 如果有类别条件，则提取标签数组
        if class_cond:
            label_arr = obj["arr_1"]
    # 获取当前进程的排名和总进程数
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []  # 图像缓冲区
    label_buffer = []  # 标签缓冲区
    # 无限循环以持续生成数据
    while True:
        # 每个进程只处理自己负责的数据部分
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            # 当缓冲区满时，生成一个批次
            if len(buffer) == batch_size:
                # 将缓冲区中的图像堆叠成一个 NumPy 数组，并转换为 PyTorch 张量
                batch = th.from_numpy(np.stack(buffer)).float()
                # 将像素值从 [0, 255] 归一化到 [-1, 1]
                batch = batch / 127.5 - 1.0
                # 调整维度顺序为 (N, C, H, W)
                batch = batch.permute(0, 3, 1, 2)
                # 构建结果字典
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                # 使用 yield 返回一个批次的数据
                yield res
                # 清空缓冲区
                buffer, label_buffer = [], []


def create_argparser():
    """
    创建命令行参数解析器。
    """
    # 设置默认参数
    defaults = dict(
        clip_denoised=True,  # 是否裁剪去噪结果
        num_samples=10000,  # 要生成的样本总数
        batch_size=16,  # 批处理大小
        use_ddim=False,  # 是否使用 DDIM 采样
        base_samples="",  # 低分辨率样本文件路径
        model_path="",  # 预训练模型路径
    )
    # 更新默认参数，加入超分辨率模型和扩散的默认设置
    defaults.update(sr_model_and_diffusion_defaults())
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 将字典中的默认参数添加到解析器
    add_dict_to_argparser(parser, defaults)
    return parser


# 当脚本作为主程序运行时执行
if __name__ == "__main__":
    main()
