"""
此代码最初是 Ho 等人扩散模型的 PyTorch 移植版本:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

我们添加了文档字符串、DDIM 采样以及一系列新的 beta 调度方案。
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    根据给定的名称获取一个预定义的 beta 调度方案。

    beta 调度库包含一系列在 num_diffusion_timesteps 趋于无穷大时保持相似的调度方案。
    可以添加新的 beta 调度方案，但一旦提交，不应删除或更改，以保持向后兼容性。

    :param schedule_name: 调度方案的名称 (例如 "linear", "cosine")。
    :param num_diffusion_timesteps: 扩散过程的总步数 T。
    :return: 一个包含 T 个 beta 值的 numpy 数组。
    """
    if schedule_name == "linear":
        # 线性调度方案，已扩展以适用于任意数量的扩散步骤。
        # beta 从 beta_start 线性增加到 beta_end。
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        # 余弦函数的 alpha_bar 调度方案，可以产生更好的生成效果。
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    创建一个 beta 调度方案，该方案离散化给定的 alpha_t_bar 函数。 详细见OPENAI 的 Improved Diffusion论文公式16
    alpha_t_bar 函数定义了 (1-beta) 从 t = [0,1] 的累积乘积。

    :param num_diffusion_timesteps: 要生成的 beta 值的数量。
    :param alpha_bar: 一个 lambda 函数，接受一个从 0 到 1 的参数 t，
                      并产生扩散过程到该部分的 (1-beta) 的累积乘积。
    :param max_beta: 要使用的最大 beta 值；使用小于 1 的值以防止奇异性。
    :return: 一个包含 T 个 beta 值的 numpy 数组。
    """
    # 最大值为0.999
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        # 根据 alpha_bar(t) = \prod_{s=1}^t (1 - beta_s) 的定义，
        # 1 - beta_t = alpha_bar(t) / alpha_bar(t-1)
        # beta_t = 1 - alpha_bar(t) / alpha_bar(t-1)
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    模型预测的输出类型。
    """

    PREVIOUS_X = enum.auto()  # 模型预测 x_{t-1}
    START_X = enum.auto()     # 模型预测 x_0
    EPSILON = enum.auto()     # 模型预测噪声 epsilon (最常用的方式)


class ModelVarType(enum.Enum):
    """
    模型预测的方差的类型。

    添加了 LEARNED_RANGE 选项，允许模型预测介于 FIXED_SMALL 和 FIXED_LARGE
    之间的值，使其任务更容易。
    """

    LEARNED = enum.auto()         # 模型学习方差 (预测一个对数值)
    FIXED_SMALL = enum.auto()     # 使用固定的较小方差 (后验方差)
    FIXED_LARGE = enum.auto()     # 使用固定的较大方差 (beta_t)
    LEARNED_RANGE = enum.auto()   # 模型学习一个插值系数，在固定的小方差和固定的大方差之间进行插值


class LossType(enum.Enum):
    """
    损失函数的类型
    """
    MSE = enum.auto()  # 使用原始的均方误差损失 (在学习方差时也使用 KL 散度)
    RESCALED_MSE = (
        enum.auto()
    )  # 使用原始的均方误差损失 (在学习方差时使用 RESCALED_KL)
    KL = enum.auto()  # 使用变分下界 (VLB) 作为损失
    RESCALED_KL = enum.auto()  # 类似于KL，但重新缩放以估计完整的 VLB

    def is_vb(self):
        """检查损失类型是否为基于变分下界 (Variational Bound) 的。"""
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    用于训练和采样扩散模型的工具类。

    :param betas: 一个一维 numpy 数组，包含每个扩散时间步的 beta 值。
    :param model_mean_type: 一个 ModelMeanType 枚举，确定模型的输出内容。是预测噪声、均值还是X_0
    :param model_var_type: 一个 ModelVarType 枚举，确定方差的输出方式。方差是可学习还是固定的，若是可学习的是直接预测方差还是预测方差线性加权的权重
    :param loss_type: 一个 LossType 枚举，确定要使用的损失函数。
    :param rescale_timesteps: 如果为 True，则将浮点类型的时间步传递给模型，
                              使其始终像原始论文中那样缩放到 (0 到 1000)。
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # 为了精度，使用 float64。
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # alpha_bar_t, 即 \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # \bar{\alpha}_{t-1}
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0) # \bar{\alpha}_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # 计算前向过程 q(x_t | x_{t-1}) 等所需的常量
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # 计算后验分布 q(x_{t-1} | x_t, x_0) 所需的常量，这个后验是高斯分布，其方差和均值系数可以预先计算
        # 后验分布真实方差
        self.posterior_variance = ( 
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 对方差取对数并裁剪。用截断防止第一项为0，因为在扩散链的开始处，后验方差为 0。
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # 后验分布均值的两个系数，可以表示为 coef1 * x_0 + coef2 * x_t
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # q是真实分布，p是神经网络分布

    def q_mean_variance(self, x_start, t):
        """
        根据x_0和t算出x_t的均值和方差。IDDPM公式8
        该函数计算出的均值和方差不是为了生成 x_t，而是为了 定义模型在反向（去噪）过程中应该学习的目标。

        -----

        获取前向过程的分布 q(x_t | x_0)。
        根据公式 $$ q ( x _ { t } | x _ { 0 } ) = \mathcal { N } ( x _ { t } ; \sqrt { \bar { \alpha } _ { t } } x _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } )$$

        :param x_start: 形状为 [N x C x ...] 的无噪声输入张量 x_0。
        :param t: 扩散步数 t。
        :return: 一个元组 (mean, variance, log_variance)，所有张量的形状都与 x_start 相同。
        """
        mean = ( # 均值 \sqrt { \bar { \alpha } _ { t } } x _ { 0 }
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape) # 方差 1 - \bar { \alpha } _ { t }
        log_variance = _extract_into_tensor( # 对数方差
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散生成带噪音的x_t

        -----

        但是该函数并没有直接调用q_mean_variance计算均值和方差，而是对公式8使用重参数方法计算均值和方差

        :param x_start: 初始数据批次 x_0。
        :param t: 扩散步数 t。
        :param noise: 如果指定，则使用提供的高斯噪声。
        :return: x_start 的加噪版本 x_t。
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        根据x_0、x_t、t计算真实后验分布q(x_{t-1} | x_t, x_0)的均值和方差
        $$q ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) = \mathcal { N } ( x _ { t - 1 } ; \tilde { \mu } ( x _ { t } , x _ { 0 } ) , \tilde { \beta } _ { t } \mathbf { I } )$$
        $\begin{align*}\tilde{\mu}_t(x_t,x_0):=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t\end{align*}$
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

#############################

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        利用模型根据x_t预测出x_{t-1}的分布（均值和方差）。

        应用模型来获取 p(x_{t-1} | x_t) 的均值和方差，以及对初始 x_0 的预测。

        :param model: 模型，它接受一个信号和一批时间步作为输入。
        :param x: 在时间 t 的张量 [N x C x ...]，即 x_t。
        :param t: 一个一维的时间步张量。
        :param clip_denoised: 如果为 True，将去噪后的信号裁剪到 [-1, 1] 范围内。
        :param denoised_fn: 如果不是 None，一个函数，在采样前应用于 x_start 预测。在 clip_denoised 之前应用。
        :param model_kwargs: 如果不是 None，一个包含额外关键字参数的字典，
                             传递给模型。这可以用于条件生成。
        :return: 一个包含以下键的字典:
                 - 'mean': 模型均值输出 (p(x_{t-1} | x_t) 的均值)。
                 - 'variance': 模型方差输出 (p(x_{t-1} | x_t) 的方差)。
                 - 'log_variance': 'variance' 的对数。
                 - 'pred_xstart': 对 x_0 的预测。
        """
        # 初始化模型关键字参数字典
        if model_kwargs is None:
            model_kwargs = {}

        # 获取输入张量的批次大小和通道数
        B, C = x.shape[:2]
        # 确保时间步张量的形状与批次大小匹配
        assert t.shape == (B,)
        # 使用模型预测当前时间步的输出，并对时间步进行缩放处理
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # 根据 model_var_type 确定方差计算方式
        ## --- 可学习方差-start---
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # 对于学习型方差，model_output形状为(B, C * 2, H, W)，C * 2 的通道维度包含两部分：前C个通道为均值，后C个通道为方差
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            # 将模型输出分割为均值部分和方差部分
            model_output, model_var_values = th.split(model_output, C, dim=1) # 若是直接预测方差，模型输出model_var_values就是方差（对数方差）；若是预测方差范围，模型输出model_var_values是插值系数。
            
            if self.model_var_type == ModelVarType.LEARNED:
                # 直接预测方差
                model_log_variance = model_var_values # 模型预测的方差是对数方差。模型一般不会直接输出方差 σ²，因为方差必须大于等于0，而神经网络的输出可以是任意实数，直接预测很难约束。在这种情况下，model_var_values 通常会被解释为对数方差（log variance），即 log(σ²)。如何得到最终方差：代码会通过取指数 th.exp(model_var_values) 来得到一个永远为正的方差值 σ²。
                model_variance = th.exp(model_log_variance) # 预测的对数方差取指数
            else: # LEARNED_RANGE IDDPM论文公式14
                # 预测方差范围。在最小和最大对数方差之间进行插值
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # model_var_values 的范围是 [-1, 1]，对应于 [min_log, max_log]。
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        ## --- 可学习方差-end---
        
        ## --- 不可学习方差-start---
        else: # FIXED_SMALL or FIXED_LARGE
            # 使用固定方差。
            model_variance, model_log_variance = { # 根据模型配置 self.model_var_type，从字典中选择一套预先定义好的方差和对数方差数组（所有时刻的）。
                ModelVarType.FIXED_LARGE: ( # $\beta_t$
                    # 对于 fixed_large，我们这样设置初始（对数）方差能够实现更好的对数似然，即能更好地拟合训练数据的整体分布
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: ( # $\tilde{\beta}_t$
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            # 从上一步选定的完整方差数组中，根据当前批次的时间步 t，提取出对应的方差值
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        ## --- 不可学习方差-end---

        # 定义处理 x_start 预测的内部函数
        def process_xstart(x):
            # 如果提供了自定义的去噪函数，先应用该函数
            if denoised_fn is not None:
                x = denoised_fn(x)
            # 如果启用裁剪，将值限制在 [-1, 1] 范围内
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # 根据 model_mean_type 确定计算x_{t-1}均值和x_0的方法。
        ## 模型预测输出有三种：x_{t-1}的均值、x_0、噪声，通过这三种内容可以计算出均值和x_0。
        ### 对于均值，模型的输出直接就是x_{t-1}的均值，然后通过x_{t-1}和x_t计算出x_0
        ### 对于x_0，模型的输出直接就是x_0，然后通过x_t和x_0计算出均值
        ### 对于噪声，通过x_t和噪声计算出x_0，然后通过x_t和x_0计算出均值
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # 模型直接预测 x_{t-1}的均值。并且额外计算出x_0,在训练中用不到，在评估中可以用到。
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            # 模型输出就是x_{t-1}均值
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                # 模型直接预测 x_0 
                pred_xstart = process_xstart(model_output)
            else: # EPSILON
                # 模型预测噪声。需要从中推导出 x_0
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            # 使用预测的 x_0 和当前的 x_t 来计算后验x_{t-1}均值，作为模型的均值输出
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        # 确保所有输出张量的形状与输入一致
        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        # 返回包含均值、方差、对数方差和 x_0 预测的字典
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        根据 x_t 和预测的噪声 eps，计算预测的 x_0。
        这是 x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon 的逆运算。
        """
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        根据 x_t 和预测的 x_{t-1}，计算预测的 x_0。
        这是后验均值公式 \tilde{\mu}_t(x_t, x_0) = coef1 * x_0 + coef2 * x_t 的逆运算。
        """
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        根据 x_t 和预测的 x_0，计算预测的噪声 eps。
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        """
        如果 rescale_timesteps 为 True，则将时间步 t 缩放到 [0, 1000] 范围。
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        对mean进行修正
        给定一个计算条件对数概率梯度 grad(log(p(y|x))) 的函数 cond_fn，
        计算上一步的条件均值。我们希望以 y 为条件。

        这使用了 Sohl-Dickstein 等人 (2015) 的条件化策略。
        新的均值 = 原始均值 + 方差 * 梯度
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        计算如果模型的得分函数被 cond_fn 条件化后，p_mean_variance 的输出会是什么。

        有关 cond_fn 的详细信息，请参见 condition_mean()。

        与 condition_mean() 不同，这里使用了 Song 等人 (2020) 的条件化策略。
        它直接修改预测的噪声 eps。
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        # 新的噪声 = 原始噪声 - sqrt(1-alpha_bar) * 梯度
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        # 基于新的噪声重新计算 x_start 和均值
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        从x_t采样出x_{t-1} (DDPM 采样，单步恢复，从噪声恢复出前一个样本)。

        :param model: 用于采样的模型。
        :param x: 当前的张量 x_t。
        :param t: 时间步 t 的值。
        :param clip_denoised: 如果为 True，将 x_start 预测裁剪到 [-1, 1]。
        :param denoised_fn: 如果不是 None，一个在采样前应用于 x_start 预测的函数。
        :param cond_fn: 如果不是 None，这是一个梯度函数，其作用类似于模型，用于引导生成。
        :param model_kwargs: 如果不是 None，一个包含额外关键字参数的字典，
                             传递给模型。这可以用于条件生成。
        :return: 一个包含以下键的字典:
                 - 'sample': 来自模型的随机样本 x_{t-1}。
                 - 'pred_xstart': 对 x_0 的预测。
        """
        # 得到x_{t-1}的均值、方差、对数方差、x_0的预测值
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # 有了x_{t-1}的均值和方差后，就是有了x_{t-1}的分布。然后通过重参化技巧从分布中采样一个x_{t-1}样本
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 当 t == 0 时不加噪声
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        # 采样公式: x_{t-1} = mean + sqrt(variance) * noise
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        从模型生成样本，循环p_sample (DDPM 采样循环)。

        :param model: 模型模块。
        :param shape: 样本的形状, (N, C, H, W)。
        :param noise: 如果指定，从编码器采样的噪声。应与 `shape` 具有相同的形状。
        :param clip_denoised: 如果为 True，将 x_start 预测裁剪到 [-1, 1]。
        :param denoised_fn: 如果不是 None，一个在采样前应用于 x_start 预测的函数。
        :param cond_fn: 如果不是 None，这是一个梯度函数，其作用类似于模型。
        :param model_kwargs: 如果不是 None，一个包含额外关键字参数的字典，
                             传递给模型。这可以用于条件生成。
        :param device: 如果指定，创建样本的设备。如果未指定，则使用模型参数的设备。
        :param progress: 如果为 True，显示 tqdm 进度条。
        :return: 一批不可微分的样本。
        """
        final = None
        # 迭代调用 p_sample_loop_progressive
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample # 保存每个中间结果
        return final["sample"] # 返回最终的样本

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        从模型生成样本，并从每个扩散时间步产生中间样本。

        参数与 p_sample_loop() 相同。
        返回一个字典的生成器，每个字典是 p_sample() 的返回值。
        """
        # 设备检测和形状验证
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        # 噪声初始化：如果没有提供噪声，从纯噪声 x_T 开始
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) # 从纯噪声 x_T 开始
        
        # 时间步序列生成
        indices = list(range(self.num_timesteps))[::-1] # 从 T-1 到 0

        if progress:
            # 延迟导入，这样我们就不依赖于 tqdm。
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices: # 从 T-1 到 0
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out # 生成器：返回中间结果
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        使用 DDIM 从模型中单步采样 x_{t-1}。

        用法与 p_sample() 相同。
        :param eta: DDIM 的参数，控制采样的随机性。eta=0.0 对应确定性采样 (DDIM)，eta=1.0 对应随机采样 (DDPM)。
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 通常我们的模型输出 epsilon，但以防我们使用了 x_start 或 x_prev 预测，我们重新推导epsilon
        # 
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # DDIM 论文中的方程 12。
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 当 t == 0 时不加噪声
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        使用 DDIM 逆向 ODE 从模型中采样 x_{t+1}。
        这用于将真实图像编码为潜在表示。
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # 通常我们的模型输出 epsilon，但我们重新推导它
        # 以防我们使用了 x_start 或 x_prev 预测。
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # 方程 12 的逆向形式
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用 DDIM 从模型生成样本。

        用法与 p_sample_loop() 相同。
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用 DDIM 从模型中采样，并从每个 DDIM 时间步产生中间样本。

        用法与 p_sample_loop_progressive() 相同。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # 延迟导入，这样我们就不依赖于 tqdm。
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]


    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        计算在单个时间步 t 上的损失（VLB）

        bpd：结果的单位是比特每维度，方便与其他论文中的结果进行公平比较。

        :return: 一个包含以下键的字典:
                 - 'output': 一个形状为 [N] 的张量，包含 NLLs 或 KLs。
                 - 'pred_xstart': x_0 的预测。
        """
        # 真实的后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        # 模型预测的分布 p(x_{t-1} | x_t) 的均值和方差
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        # 根据上述的真实和预测的均值、方差计算两个高斯分布之间的 KL 散度
        # 对应着L[t-1]损失函数 IDDPM公式6
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0) # 转换为比特单位

        # 对应着L[0]损失函数 IDDPM公式5
        decoder_nll = -discretized_gaussian_log_likelihood( # 计算解码器的负对数似然 (NLL)
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0) # 转换为比特单位

        # 在第一个时间步 (t=0) 返回解码器 NLL，否则返回 KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # t=0时刻，用离散的高斯分布去计算似然
        # t>0时刻，直接用KL散度
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        计算单个时间步的训练损失。

        :param model: 用于评估损失的模型。
        :param x_start: 形状为 [N x C x ...] 的输入张量 x_0。
        :param t: 一批时间步索引。
        :param model_kwargs: 如果不是 None，一个包含额外关键字参数的字典，
                             传递给模型。这可以用于条件生成。
        :param noise: 如果指定，则使用特定的高斯噪声。
        :return: 一个包含键 "loss" 的字典，其值为形状为 [N] 的张量。
                 某些均值或方差设置可能还会有其他键。
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        # 1. 从 x_0 和 t 生成加噪图像 x_t
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        # 2. 根据损失类型计算损失
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            # 使用变分下界(KL)作为损失
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # 使用均方误差MSE作为损失
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # 使用变分界学习方差，但不让它影响我们的均值预测。
                # `detach()` 用于阻止梯度流向均值预测部分。
                # 预测方差时，mean和var一起计算KL损失，但是只有var能优化，mean通过后面的MSE进行优化。
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # 除以 1000 以与初始实现等效。
                    # 如果没有 1/1000 的因子，VB 项会损害 MSE 项。
                    terms["vb"] *= self.num_timesteps / 1000.0

            # 确定 MSE 的目标
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        获取变分下界的先验 KL 项，以 bits-per-dim 为单位。
        这个项是 KL(q(x_T|x_0) || p(x_T))，其中 p(x_T) 是标准正态分布。

        这个项不含参数，无法优化，因为它只依赖于编码器（前向过程）。

        :param x_start: 形状为 [N x C x ...] 的输入张量。
        :return: 一批 [N] 个 KL 值（以比特为单位），每个批次元素一个。
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        计算整个变分下界 (VLB)，从T到0把所有的loss都算出来
        训练中不会用到，用于评估模型性能。

        :param model: 用于评估损失的模型。
        :param x_start: 形状为 [N x C x ...] 的输入张量。
        :param clip_denoised: 如果为 True，裁剪去噪后的样本。
        :param model_kwargs: 如果不是 None，一个包含额外关键字参数的字典，
                             传递给模型。这可以用于条件生成。

        :return: 一个包含以下键的字典:
                 - total_bpd: 每个批次元素的总变分下界。
                 - prior_bpd: 下界中的先验项。
                 - vb: 一个 [N x T] 的张量，包含下界中的各项。
                 - xstart_mse: 一个 [N x T] 的张量，包含每个时间步的 x_0 MSE。
                 - mse: 一个 [N x T] 的张量，包含每个时间步的 epsilon MSE。
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # 计算当前时间步的 VLB 项
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    把arr的第timesteps个取出来，并且形状等于broadcast_shape

    :param arr: 一维 numpy 数组。
    :param timesteps: 一个张量，包含要提取的数组索引。
    :param broadcast_shape: 一个 K 维的更大形状，其批次维度，等于 timesteps 的长度。
    :return: 一个形状为 [batch_size, 1, ...] 的张量，该形状有 K 个维度。
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
