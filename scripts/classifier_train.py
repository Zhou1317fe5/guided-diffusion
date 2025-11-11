"""
在 ImageNet 上训练一个加噪图像分类器。
这个脚本的核心是训练一个分类器，该分类器专门用于对经过扩散过程加噪的图像进行分类。
这是实现分类器引导（Classifier Guidance）技术的前提。

为什么要在加噪图像上训练分类器？
为了在采样过程中有效引导，分类器必须能够识别出在任意噪声水平（即任意时间步 t）下的图像内容。
如果只用干净图像训练，分类器在面对反向扩散过程中的中间噪声图像时将无法提供有意义的梯度。
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main():
    """
    主训练函数，负责整个分类器的训练流程。
    训练流程包括：
    1. 初始化分布式环境和日志记录器。
    2. 创建分类器模型和扩散过程实例（扩散过程用于对图像加噪）。
    3. 设置 `schedule_sampler`，它决定了在训练时如何为图像随机选择加噪的时间步 `t`。
    4. 配置混合精度训练器 `MixedPrecisionTrainer` 和分布式数据并行 `DDP`。
    5. 加载训练和验证数据集。
    6. 创建优化器（AdamW）。
    7. 进入主训练循环，迭代执行训练、验证、日志记录和模型保存。
    """
    args = create_argparser().parse_args()

    # 初始化分布式训练环境
    dist_util.setup_dist()
    # 配置日志记录器
    logger.configure()

    logger.log("creating model and diffusion...")
    # 创建分类器模型和扩散过程实例
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev()) # 将模型移动到指定设备
    # 如果设置为在加噪图像上训练
    if args.noised:
        # 创建一个 schedule sampler，用于在训练过程中为每个样本随机采样一个时间步 t
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    # 如果指定了恢复点，则从中加载模型
    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # 在分布式训练中同步所有进程的模型参数
    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    # 创建混合精度训练器
    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    # 使用 DistributedDataParallel (DDP) 包装模型以支持分布式训练
    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    # 加载训练数据集
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
    )
    # 如果提供了验证数据集路径，则加载验证数据集
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    # 创建 AdamW 优化器
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    # 如果指定了恢复点，则也加载优化器的状态
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        """
        执行一个训练或验证批次的前向传播、反向传播和日志记录。

        Args:
            data_loader: 数据加载器，用于提供数据批次。
            prefix: 日志前缀，通常是 "train" 或 "val"。

        核心逻辑:
        1. 从数据加载器中获取一批干净的图像和对应的标签。
        2. **核心步骤**: 使用 `schedule_sampler` 随机采样时间步 `t`。
        3. **核心步骤**: 调用 `diffusion.q_sample(batch, t)` 对干净的图像 `batch` 施加 `t` 步的噪声，生成训练样本。
           这是至关重要的一步，因为分类器被训练来识别**加噪后**的图像内容，
           这样它才能在生成过程中引导去噪方向。
        4. 将加噪后的图像和时间步 `t` 一同输入分类器模型，获得预测 `logits`。
        5. 计算预测与真实标签之间的交叉熵损失。
        6. 记录损失和准确率等指标。
        7. 执行反向传播和梯度更新。
        """
        # 从数据加载器中获取一个批次的数据
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # 如果设置为在加噪图像上训练
        if args.noised:
            # 随机采样时间步 t
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            # 使用扩散过程的 q_sample 方法对干净图像 batch 施加 t 步噪声
            batch = diffusion.q_sample(batch, t)
        else:
            # 如果不加噪，则时间步 t 为 0
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        # 为了处理大批量数据，可以将其拆分为多个微批次（microbatch）
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            # 前向传播：将加噪后的图像和时间步输入模型，得到预测 logits
            logits = model(sub_batch, timesteps=sub_t)
            # 计算交叉熵损失
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            # 记录损失和准确率
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
            log_loss_dict(diffusion, sub_t, losses)
            del losses

            # 计算平均损失
            loss = loss.mean()
            # 如果需要计算梯度，则执行反向传播
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    # 主训练循环
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        # 如果设置了学习率退火，则更新学习率
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        
        # 执行一个训练步
        forward_backward_log(data)
        # 优化器执行一步更新
        mp_trainer.optimize(opt)

        # 定期在验证集上进行评估
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync(): # 在评估时禁用 DDP 的梯度同步
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        
        # 定期记录日志
        if not step % args.log_interval:
            logger.dumpkvs()
        
        # 定期保存模型
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    # 训练结束后，最后保存一次模型
    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    """
    设置线性退火的学习率。
    :param opt: 优化器。
    :param base_lr: 基础学习率。
    :param frac_done: 训练完成的比例 (从 0 到 1)。
    """
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    """
    保存模型和优化器状态。
    只在主进程 (rank 0) 中执行。
    """
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    """
    计算 Top-K 准确率。
    :param logits: 模型的预测输出。
    :param labels: 真实标签。
    :param k: Top-K 中的 K 值。
    :param reduction: "mean" 表示返回平均准确率，"none" 表示返回每个样本的准确率。
    """
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    """
    将一个大批次拆分成多个微批次。
    """
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    """
    创建命令行参数解析器并设置默认值。
    """
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,  # -1 表示不使用微批次
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
