"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import numpy as np
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    """
    主函数，负责执行整个评估流程。
    这个脚本的核心是评估一个训练好的扩散模型在给定数据集上的性能。
    它通过计算数据的负对数似然（NLL）来实现，结果通常以“每维度比特数”（Bits Per Dimension, BPD）为单位报告。

    评估流程包括：
    1. 加载预训练的扩散模型。
    2. 加载需要进行评估的数据集（例如测试集）。
    3. 调用 run_bpd_evaluation 函数来执行核心的评估计算。
    """
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    """
    执行 BPD (Bits Per Dimension) 评估。

    BPD 是一种衡量生成模型拟合真实数据分布能力的指标。它表示模型压缩每个数据维度所需的平均比特数。
    BPD 值越低，说明模型为真实数据分配的概率越高，模型的生成能力就越好。

    计算过程如下：
    1. 遍历数据集中的所有样本。
    2. 对于每个批次的样本，调用 `diffusion.calc_bpd_loop` 函数。这个函数是计算 BPD 的核心，
       它通过计算变分下界 (Variational Lower Bound, VLB) 来近似真实的数据对数似然。
    3. `calc_bpd_loop` 返回一个包含多个度量的字典，包括 VLB 的各个组成部分 (`vb`, `mse` 等)
       以及最终的 `total_bpd`。
    4. 在分布式计算环境中，需要对所有进程计算出的 BPD 值进行 `all_reduce` 操作，以获得全局平均值。
    5. 累积所有批次的结果，计算最终的平均 BPD。

    :param model: 要评估的扩散模型。
    :param diffusion: 扩散过程的辅助对象，包含了 BPD 计算的逻辑。
    :param data: 数据加载器，提供评估用的数据批次。
    :param num_samples: 要评估的样本总数。
    :param clip_denoised: 是否在计算过程中对去噪结果进行裁剪。
    """
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())
        num_complete += dist.get_world_size() * batch.shape[0]

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")

    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
