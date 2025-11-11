"""
各种基于似然的损失函数的辅助工具。这些函数移植自原始的
Ho et al. 扩散模型代码库：
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py

这些函数主要用于扩散模型中的损失计算，包括KL散度计算、
正态分布累积分布函数的近似计算，以及离散化高斯分布的对数似然计算。
"""

import numpy as np
import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个高斯分布之间的KL散度（Kullback-Leibler散度）。
    
    KL散度是衡量两个概率分布差异的重要指标，在扩散模型中
    用于计算预测分布与真实分布之间的差异，是训练过程中的
    关键损失函数之一。

    :param mean1: 第一个高斯分布的均值张量
    :param logvar1: 第一个高斯分布的对数方差张量 (log(σ₁²))
    :param mean2: 第二个高斯分布的均值张量
    :param logvar2: 第二个高斯分布的对数方差张量 (log(σ₂²))
    :return: 两个分布之间的KL散度张量，形状与输入张量相同
    """
    tensor = None
    # 遍历所有输入参数，找到第一个张量对象作为设备张量
    # 这样可以确保所有后续计算都在同一个设备上（CPU或GPU）
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    # 确保至少有一个参数是张量，否则无法确定计算设备
    assert tensor is not None, "at least one argument must be a Tensor"


    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # KL散度的计算公式：KL(N(μ1,σ1²) || N(μ2,σ2²))
    # = 0.5 * (log(σ2²) - log(σ1²) - 1 + (σ1² + (μ1-μ2)²)/σ2²)
    # 转换为对数方差的表达形式
    return 0.5 * (
        -1.0                                          # 常数项 -1
        + logvar2                                     # log(σ2²)
        - logvar1                                     # -log(σ1²)
        + th.exp(logvar1 - logvar2)                  # exp(log(σ1²) - log(σ2²)) = σ1²/σ2²
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2) # (μ1-μ²)² * exp(-log(σ2²)) = (μ1-μ2)²/σ2²
    )


def approx_standard_normal_cdf(x):
    """

    """
    # 实现Hastings近似公式
    # √(2/π) 是标准化常数
    # 0.044715 是拟合参数，用于提高近似精度
    # x³ 项提供了三阶修正，显著提高近似精度
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    计算数据 x 在一个离散化高斯分布下的对数似然。根据模型预测的均值（means）和方差（log_scales），我们观察到的真实数据 x 出现的概率有多大？
    """

    assert x.shape == means.shape == log_scales.shape, "所有输入张量必须具有相同的形状"
    
    # 减去均值
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)

    # 将[-1, 1]分成255个bins，最右边的CDF记为1，最左边的CDF记为0
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    
    # 用小范围的CDF之差来表示PDF
    cdf_delta = cdf_plus - cdf_min

    # 考虑到两个极限的地方，这里用到了两个where
    log_probs = th.where(
        x < -0.999,  # 接近下边界的情况，使用上尾概率
        log_cdf_plus,
        th.where(
            x > 0.999,  # 接近上边界的情况，使用下尾概率
            log_one_minus_cdf_min,
            th.log(cdf_delta.clamp(min=1e-12))  # 中间区域使用精确概率
        ),
    )
    
    # 验证输出形状的正确性
    assert log_probs.shape == x.shape, "输出张量形状必须与输入相同"
    return log_probs
