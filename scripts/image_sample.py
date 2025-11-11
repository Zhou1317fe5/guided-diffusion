"""
从模型生成大批量图像样本，并将它们保存为一个大的Numpy数组。
这可用于为FID评估生成样本。
"""

# 导入 argparse 用于解析命令行参数。
import argparse
# 导入 os 用于与操作系统交互，例如文件路径操作。
import os

# 导入 numpy 用于高效的数值计算。
import numpy as np
# 导入 torch 用于深度学习任务。
import torch as th
# 导入 torch.distributed 用于分布式计算。
import torch.distributed as dist

# 从 guided_diffusion 库中导入分布式工具和日志记录器。
from guided_diffusion import dist_util, logger
# 从 guided_diffusion.script_util 中导入一些工具函数和默认配置。
from guided_diffusion.script_util import (
    NUM_CLASSES,  # 导入类别总数。
    model_and_diffusion_defaults,  # 导入模型和扩散过程的默认参数。
    create_model_and_diffusion,  # 导入创建模型和扩散过程的函数。
    add_dict_to_argparser,  # 导入将字典添加到 argparse 解析器的函数。
    args_to_dict,  # 导入将 argparse 参数转换为字典的函数。
)


def main():
    """
    主函数，执行图像采样过程。
    """
    # 创建参数解析器并解析命令行参数。
    args = create_argparser().parse_args()

    # 设置分布式环境。
    dist_util.setup_dist()
    # 配置日志记录器。
    logger.configure()

    # 记录日志：正在创建模型和扩散过程。
    logger.log("creating model and diffusion...")
    # 创建模型和扩散过程实例。
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # 从指定路径加载模型权重。
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # 将模型移动到指定的计算设备（CPU或GPU）。
    model.to(dist_util.dev())
    # 如果使用 FP16 精度，则转换模型。
    if args.use_fp16:
        model.convert_to_fp16()
    # 将模型设置为评估模式。
    model.eval()

    # 记录日志：开始采样。
    logger.log("sampling...")
    # 初始化用于存储所有生成图像的列表。
    all_images = []
    # 初始化用于存储所有标签的列表。
    all_labels = []
    # 循环直到生成的样本数量达到要求。
    while len(all_images) * args.batch_size < args.num_samples:
        # 初始化模型关键字参数字典。
        model_kwargs = {}
        # 如果使用类别条件。
        if args.class_cond:
            # 随机生成一批类别标签。
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            # 将类别标签添加到模型关键字参数中。
            model_kwargs["y"] = classes
        # 根据是否使用 DDIM 选择采样函数。
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # 调用采样函数生成一批样本。
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),  # 样本形状
            clip_denoised=args.clip_denoised,  # 是否裁剪去噪后的结果
            model_kwargs=model_kwargs,  # 模型关键字参数
        )
        # 将样本从 [-1, 1] 范围转换到 [0, 255] 范围。
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # 调整样本维度顺序，从 (N, C, H, W) 变为 (N, H, W, C)。
        sample = sample.permute(0, 2, 3, 1)
        # 确保张量在内存中是连续的。
        sample = sample.contiguous()

        # 为分布式环境中的每个进程创建一个零张量列表，用于收集样本。
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # 在所有进程之间收集样本。
        dist.all_gather(gathered_samples, sample)  # NCCL后端不支持gather操作
        # 将收集到的样本添加到 all_images 列表中。
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # 如果使用类别条件。
        if args.class_cond:
            # 为分布式环境中的每个进程创建一个零张量列表，用于收集标签。
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            # 在所有进程之间收集标签。
            dist.all_gather(gathered_labels, classes)
            # 将收集到的标签添加到 all_labels 列表中。
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # 记录日志：已创建的样本数量。
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # 将所有图像样本拼接成一个大的 numpy 数组。
    arr = np.concatenate(all_images, axis=0)
    # 截取到所需的样本数量。
    arr = arr[: args.num_samples]
    # 如果使用类别条件。
    if args.class_cond:
        # 将所有标签拼接成一个大的 numpy 数组。
        label_arr = np.concatenate(all_labels, axis=0)
        # 截取到所需的样本数量。
        label_arr = label_arr[: args.num_samples]
    # 只有主进程（rank 0）执行保存操作。
    if dist.get_rank() == 0:
        # 构建输出文件名的形状字符串。
        shape_str = "x".join([str(x) for x in arr.shape])
        # 构建输出文件的完整路径。
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        # 记录日志：保存文件的路径。
        logger.log(f"saving to {out_path}")
        # 如果使用类别条件，则同时保存图像和标签。
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        # 否则只保存图像。
        else:
            np.savez(out_path, arr)

    # 在所有进程之间设置一个屏障，确保所有进程都执行到这里。
    dist.barrier()
    # 记录日志：采样完成。
    logger.log("sampling complete")


def create_argparser():
    """
    创建并返回一个命令行参数解析器。
    """
    # 定义默认参数。
    defaults = dict(
        clip_denoised=True,  # 是否裁剪去噪结果
        num_samples=10000,  # 生成的样本总数
        batch_size=16,  # 每批次的样本数
        use_ddim=False,  # 是否使用 DDIM 采样
        model_path="",  # 模型权重的路径
    )
    # 更新默认参数，加入模型和扩散过程的默认参数。
    defaults.update(model_and_diffusion_defaults())
    # 创建 ArgumentParser 对象。
    parser = argparse.ArgumentParser()
    # 将默认参数字典添加到解析器中。
    add_dict_to_argparser(parser, defaults)
    # 返回解析器。
    return parser


if __name__ == "__main__":
    # 如果该脚本作为主程序运行，则调用 main 函数。
    main()
