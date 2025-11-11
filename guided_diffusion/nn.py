"""
各种神经网络相关的工具函数。
"""

import math

import torch as th
import torch.nn as nn


# PyTorch 1.7 版本引入了 SiLU 激活函数，但为了兼容 PyTorch 1.5 版本，这里自定义实现。
class SiLU(nn.Module):
    """
    SiLU (Sigmoid-weighted Linear Unit) 激活函数，也称为 Swish。
    计算公式为: f(x) = x * sigmoid(x)
    """
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """
    一个自定义的 Group Normalization 层，将输入转换为 float32 进行计算，然后再转换回原始数据类型。
    这主要是为了在混合精度训练（如使用 float16）时，保证归一化计算的数值稳定性。
    Group Normalization 是一种将通道分成若干组，并在组内进行归一化的技术，
    使其性能与批大小无关，适用于批大小较小的场景。
    """
    def forward(self, x):
        # 将输入 x 转换为 float32 类型进行归一化计算，然后将结果转换回 x 的原始数据类型。
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    根据维度 `dims` (1, 2, 或 3) 创建一个 1D, 2D, 或 3D 的卷积模块。
    这是一个工厂函数，方便根据需要动态创建不同维度的卷积层。
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    创建一个线性（全连接）模块。
    这是 nn.Linear 的一个简单包装。
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    根据维度 `dims` (1, 2, 或 3) 创建一个 1D, 2D, 或 3D 的平均池化模块。
    这是一个工厂函数，方便根据需要动态创建不同维度的池化层。
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    使用指数移动平均（Exponential Moving Average, EMA）更新目标参数，使其接近源参数。
    EMA 是一种在训练过程中维护模型参数平滑副本的方法。这个平滑后的模型（EMA模型）
    通常具有更好的泛化能力，在评估和生成时能产生更稳定、更高质量的结果。

    更新公式为: target = rate * target + (1 - rate) * source

    :param target_params: 目标参数序列（EMA 模型的参数）。
    :param source_params: 源参数序列（当前训练步的模型参数）。
    :param rate: EMA 的更新率（越接近 1，更新越慢，模型越平滑）。
    """
    # 遍历参数并进行原地（in-place）更新
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    将一个模块的所有参数置零，并返回该模块。
    这通常用于U-Net的输出层，确保在训练开始时，模型的输出对最终结果没有影响，
    让模型从一个确定的状态开始学习。
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    将一个模块的所有参数乘以一个缩放因子，并返回该模块。
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    计算一个张量在所有非批次维度（non-batch dimensions）上的平均值。
    例如，对于一个形状为 [N, C, H, W] 的张量，它会计算 C, H, W 维度上的平均值，
    最终返回一个形状为 [N] 的张量。
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    创建一个标准的归一化层。
    在这里，标准归一化层被定义为 GroupNorm32，其中组的数量固定为 32。

    :param channels: 输入通道数。
    :return: 一个用于归一化的 nn.Module。
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建正弦（sinusoidal）时间步嵌入。
    这个函数将一个一维的时间步张量（每个批次元素一个时间步）转换为高维的向量嵌入。
    其原理与 Transformer 中的位置编码（Positional Encoding）相同。通过使用不同频率的
    sin 和 cos 函数，可以将标量时间步 `t` 映射到一个独特的、高维的向量表示，
    使得神经网络能够轻易地理解时间步信息。

    :param timesteps: 一个一维张量，包含 N 个时间步索引，每个批次元素一个。这些值可以是浮点数。
    :param dim: 输出嵌入的维度。
    :param max_period: 控制嵌入的最小频率。
    :return: 一个形状为 [N x dim] 的位置嵌入张量。
    """
    half = dim // 2
    # 计算频率，频率从 1 到 1/max_period 呈几何级数递减
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    # 将时间步与频率相乘
    args = timesteps[:, None].float() * freqs[None]
    # 计算 sin 和 cos 嵌入
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    # 如果维度是奇数，则在末尾填充一个零
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    评估一个函数，但不缓存中间激活值。这可以减少内存消耗，但代价是在反向传播时需要额外的计算。
    这是一种典型的“用计算换内存”的策略，也称为梯度检查点（Gradient Checkpointing）。
    当模型非常大，导致中间激活值占用过多显存时，这个技术非常有用。

    :param func: 要评估的函数。
    :param inputs: 传递给 `func` 的参数序列。
    :param params: `func` 依赖但没有作为显式参数传入的参数序列。
    :param flag: 如果为 False，则禁用梯度检查点，正常执行函数。
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        # 如果不使用 checkpoint，则直接调用函数
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    """
    torch.autograd.Function 的一个自定义实现，用于执行梯度检查点。
    """
    @staticmethod
    def forward(ctx, run_function, length, *args):
        """
        在前向传播中，我们只运行函数并保存必要的上下文信息，但不保存计算图。
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        # 在不追踪梯度的情况下运行前向传播，以节省内存
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        """
        在反向传播中，我们重新计算前向传播，以构建计算图，然后计算梯度。
        """
        # 将输入张量设置为需要梯度，以便进行反向传播
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # 开启梯度计算
        with th.enable_grad():
            # 修复一个 bug：run_function 中的第一个操作可能会原地修改张量存储，
            # 这对于 detach()'d 的张量是不允许的。创建一个浅拷贝可以避免这个问题。
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        
        # 计算输入和参数的梯度
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        # 清理上下文，释放内存
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        # 返回的梯度需要与 forward 的输入一一对应
        return (None, None) + input_grads
