"""
与 image_sample.py 类似，但使用一个在加噪图像上训练的分类器来引导采样过程，
以生成更逼真的、符合特定类别的图像。
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    """
    主函数，执行分类器引导的图像采样流程。
    流程如下：
    1. 解析命令行参数。
    2. 设置分布式训练环境。
    3. 加载预训练的扩散模型（U-Net），该模型用于在逆向扩散过程中逐步去噪。
    4. 加载一个在加噪图像上训练过的分类器，该分类器能够识别含噪图像的内容。
    5. 定义分类器引导函数 `cond_fn`，这是实现引导的核心。
    6. 定义模型预测函数 `model_fn`，它包装了扩散模型。
    7. 进入主采样循环，直到生成指定数量的图像：
       a. 随机生成一批目标类别 `y`，并将其放入 `model_kwargs` 中。
       b. 根据是否使用 DDIM，选择 `p_sample_loop` 或 `ddim_sample_loop` 作为采样器。
       c. 调用采样器函数，传入 `model_fn`、`cond_fn` 和 `model_kwargs`。
          在每个采样步骤中，`cond_fn` 会计算梯度来“引导”去噪方向，使其生成的图像更符合目标类别 `y`。
       d. 收集并后处理生成的图像样本（从 [-1, 1] 范围转换到 [0, 255] 范围）。
    8. 将所有进程生成的图像和标签样本聚合起来。
    9. 如果是主进程 (rank 0)，则将最终的图像和标签数组保存为 `.npz` 文件。
    """
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    # 加载预训练的扩散模型
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    # 加载在加噪图像上训练的分类器
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        """
        分类器引导的条件函数 (conditioning function)。
        这个函数在每个采样步骤都会被调用，其目的是计算一个梯度，用于引导扩散过程。

        原理：
        1. 输入当前含噪图像 `x`、时间步 `t` 和目标类别 `y`。
        2. 使用分类器 `classifier` 预测 `x` 属于各个类别的对数概率 `log_probs`。
        3. 提取出图像属于目标类别 `y` 的对数概率 `selected`。
        4. 计算 `selected` 相对于输入图像 `x` 的梯度。这个梯度 `th.autograd.grad(selected.sum(), x_in)[0]`
           指向了对 `x` 进行微小改动，能够最大程度增加其被分类为目标类别 `y` 的概率的方向。
        5. 将该梯度乘以一个缩放因子 `classifier_scale`，得到最终的引导信号。
           这个信号会被加到扩散模型的预测噪声上，从而“推动”或“引导”下一个采样步骤的 `x_{t-1}`
           不仅符合扩散模型的学习分布，还更强烈地表现出目标类别 `y` 的特征。
        """
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            # 计算梯度并乘以引导强度因子
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        """
        一个包装函数，用于调用扩散模型。
        它确保在有类别条件时，将类别信息 `y` 传递给模型。
        """
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    # 循环生成样本，直到达到指定的数量
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        # 随机生成一批目标类别
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        
        # 选择采样器：p_sample_loop (DDPM) 或 ddim_sample_loop (DDIM)
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # 执行采样过程
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,  # 传入分类器引导函数
            device=dist_util.dev(),
        )
        # 对生成的样本进行后处理，从 [-1, 1] 映射到 [0, 255]
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # 收集来自所有分布式进程的样本
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        # 收集来自所有分布式进程的标签
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # 将所有样本和标签拼接成一个大的Numpy数组
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    
    # 主进程负责保存结果
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    """
    创建命令行参数解析器。
    """
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
