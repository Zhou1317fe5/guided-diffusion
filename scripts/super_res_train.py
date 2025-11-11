"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    """
    超分辨率模型训练的主入口函数。

    整个训练流程如下：
    1. 解析命令行参数。
    2. 设置分布式训练环境和日志记录器。
    3. 调用 sr_create_model_and_diffusion 创建超分辨率U-Net模型和高斯扩散过程实例。
    4. 调用 load_superres_data 函数准备训练数据加载器。
    5. 实例化一个通用的 TrainLoop 对象，该对象封装了整个训练循环的逻辑。
       这个脚本本身不直接实现训练循环，而是复用了 guided_diffusion/train_util.py 中定义的 TrainLoop 类。
    6. 调用 TrainLoop 对象的 run_loop() 方法，启动训练过程。
    """
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    # 创建超分辨率模型和扩散过程实例
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # 调用 load_superres_data 函数来准备训练数据
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    # 将所有配置（模型、扩散、数据、训练参数等）传递给一个通用的 TrainLoop 对象
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()  # 调用 run_loop() 方法启动训练过程


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    """
    为超分辨率任务创建并返回一个数据加载生成器。

    这个函数的核心逻辑是动态地生成低分辨率图像作为模型的条件输入。
    它首先加载高分辨率图像，然后即时(on-the-fly)地创建对应的低分辨率版本。

    :param data_dir: 训练数据集的路径。
    :param batch_size: 每个批次中的样本数量。
    :param large_size: 原始高分辨率图像的尺寸，作为训练目标。
    :param small_size: 目标低分辨率图像的尺寸，作为模型输入条件。
    :param class_cond: 是否使用类别条件。
    :return: 一个生成器，每次产出一个元组 (高分辨率图像批次, 模型关键字参数字典)。
             字典中包含了作为条件的低分辨率图像 'low_res'。
    """
    # 首先，使用通用的 load_data 函数加载一批高分辨率的图像 (large_batch)。
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    # 遍历高分辨率数据加载器
    for large_batch, model_kwargs in data:
        # 核心步骤: 使用 F.interpolate 函数，通过 'area' 模式对高分辨率图像进行下采样，
        # 动态地创建出对应的低分辨率版本 (low_res)。
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        # yield 一个元组，包含高分辨率图像 (作为训练目标 x_0) 和一个 model_kwargs 字典。
        # 这个字典中包含了关键的条件信息，即下采样后的低分辨率图像 low_res。
        yield large_batch, model_kwargs


def create_argparser():
    """
    创建一个 ArgumentParser 对象，用于解析命令行参数。
    """
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
