"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    """
    主训练函数。
    """
    # 创建参数解析器并解析命令行参数
    args = create_argparser().parse_args()

    # 设置分布式训练环境
    dist_util.setup_dist()
    # 配置日志记录器
    logger.configure()

    logger.log("creating model and diffusion...")
    # 根据参数创建U-Net模型和diffusion过程
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()) # args是很大的参数集合（包括训练、模型、diffusion等），这里只需要模型和diffusion的部分，所以通过args_to_dict从args中提取model_and_diffusion_defaults()的参数，然后将提取的参数键值对传入到create_model_and_diffusion中
    )
    # 将模型移动到指定的计算设备（CPU或GPU）
    model.to(dist_util.dev())
    # 创建一个调度采样器，用于在训练过程中动态地选择t
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # 加载数据集
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    # 初始化并运行训练循环
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
    ).run_loop()


# 命令行传参技巧
def create_argparser():
    """
    从字典中自动生成命令行传参的argument parser
    :return: 返回一个配置好默认参数的 argparse.ArgumentParser 对象。
    """
    # 定义训练脚本的默认参数
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",  # 调度采样器类型，默认为均匀采样
        lr=1e-4,  # 学习率
        weight_decay=0.0,  # 权重衰减
        lr_anneal_steps=0,  # 学习率退火步数
        batch_size=1,  # 批处理大小
        microbatch=-1,  # 微批次大小，-1表示不使用微批次
        ema_rate="0.9999",  # 指数移动平均（EMA）率
        log_interval=10,  # 日志记录间隔
        save_interval=10000,  # 模型保存间隔
        resume_checkpoint="",  # 从指定的检查点恢复训练
        use_fp16=False,  # 是否使用16位浮点数精度进行训练
        fp16_scale_growth=1e-3,  # fp16的缩放因子增长率
    )
    # 将模型和diffusion的默认参数更新到defaults字典中
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # 将字典中的所有默认参数添加到解析器中
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
