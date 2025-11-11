"""
Helpers to train with 16-bit precision.
"""
# 导入用于16位精度（FP16）训练的辅助工具。
# 混合精度训练是一种在训练深度学习模型时同时使用16位和32位浮点数（FP16和FP32）的技术。
# 它的主要优点是：
# 1. 减少内存占用：FP16数值占用的内存是FP32的一半，这使得可以训练更大的模型或使用更大的批量大小。
# 2. 加快计算速度：现代GPU（如NVIDIA的Tensor Cores）对FP16的计算速度远快于FP32。
#
# 然而，FP16的数值范围比FP32窄，更容易出现上溢（overflow）和下溢（underflow）问题。
# 为了解决这些问题，混合精度训练通常采用以下策略：
# 1. 主参数（Master Parameters）：保留一份FP32精度的模型参数副本（主参数），用于更新权重，防止因FP16精度不足导致的更新错误。
# 2. 损失缩放（Loss Scaling）：将损失值乘以一个缩放因子，以将梯度值拉升到FP16可以安全表示的范围内，从而防止梯度下溢。

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import logger

# 初始的对数损失缩放因子。
# 在混合精度训练中，损失缩放（Loss Scaling）是防止梯度下溢的关键技术。
# 如果梯度值非常小，在FP16下可能会变成0，导致模型参数无法更新。
# 通过将损失乘以一个大的缩放因子（scale factor），可以按比例放大梯度，使其进入FP16的有效表示范围。
# 这里的初始值是20，意味着初始的损失缩放因子是 2**20。
INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    # 将模型的特定层（如卷积层）的权重和偏置转换为float16（半精度）。
    # 这是应用混合精度训练的第一步，将模型的大部分计算转换为FP16以加速训练。
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    # 将模型的特定层转换回float32（单精度），是 convert_module_to_f16() 的逆操作。
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    # 创建一份FP32精度的主参数（master parameters）。
    # 在混合精度训练中，虽然前向和反向传播在FP16上进行以提高效率，
    # 但参数的更新步骤在FP32上进行，以保持数值的稳定性和精度。
    # 这个函数将模型的FP16参数复制并转换为FP32，作为优化器更新的“主”版本。
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        # 将一组参数（param_group）展平（flatten）成一个一维张量，
        # 然后转换为FP32，并重新塑形（view）为指定形状。
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True # 确保主参数可以计算梯度
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    # 将FP16模型参数的梯度复制到FP32主参数上。
    # 反向传播计算出的梯度是针对FP16模型参数的，需要将这些梯度
    # 传递给对应的FP32主参数，以便进行精确的权重更新。
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        # 同样，将一组模型参数的梯度展平，然后赋给对应的主参数梯度。
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # 将更新后的FP32主参数的值复制回FP16模型参数。
    # 在优化器更新了主参数之后，需要将这些新的、更精确的权重
    # 同步回模型参数，以用于下一次的前向传播。
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        # unflatten_master_params 将展平的主参数还原为原始模型参数的形状列表。
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            # 使用 detach().copy_() 进行原地复制，避免不必要的计算图连接。
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    # 这是一个辅助函数，用于将展平的（flattened）主参数张量
    # 还原（unflatten）成与原始参数组（param_group）结构和形状相同的张量列表。
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    # 将模型的命名参数分为两组：
    # 1. 标量和向量参数（维度 <= 1），例如偏置（bias）或层归一化（LayerNorm）的参数。
    # 2. 矩阵参数（维度 > 1），例如卷积层或全连接层的权重。
    # 这种分组可以优化处理，因为不同形状的参数在展平操作时可能需要不同的处理方式。
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1), # 展平后的形状
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1), # 展平后的形状
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    # 将主参数（master_params）保存到模型的 state_dict 中。
    # 这在保存模型 checkpoint 时非常重要，因为我们需要保存高精度的FP32参数，
    # 而不是可能存在精度损失的FP16模型参数。
    if use_fp16:
        state_dict = model.state_dict()
        # 从FP32主参数中恢复参数并更新到 state_dict
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        # 如果不使用FP16，则模型参数就是主参数，直接保存即可。
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    # 从 state_dict 加载参数来创建主参数。
    # 这在从 checkpoint 恢复训练时使用，确保加载的是高精度的FP32参数。
    if use_fp16:
        # 从 state_dict 中获取模型参数
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        # 根据加载的参数创建新的分组和形状信息
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        # 创建FP32主参数
        master_params = make_master_params(param_groups_and_shapes)
    else:
        # 如果不使用FP16，直接从 state_dict 加载参数。
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    # 清空主参数的梯度。
    # 在优化器步骤（optimizer.step()）之后调用，为下一次迭代做准备。
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    # 清空模型参数的梯度。
    for param in model_params:
        # 代码源自 PyTorch 的优化器实现，用于安全地清空梯度。
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    # 如果参数有梯度，则返回其梯度；否则返回一个形状相同的零张量。
    # 这在 `model_grads_to_master_grads` 中使用，确保所有参数都有一个有效的梯度值可以被展平。
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class MixedPrecisionTrainer:
    """
    一个封装了混合精度训练逻辑的训练器类。

    它负责管理FP32主参数、FP16模型参数、损失缩放以及它们之间的同步。
    核心思想是：
    1. 前向传播和损失计算在FP16模型上进行，以获得速度优势。
    2. 反向传播时，为了防止梯度下溢，将损失乘以一个缩放因子（loss_scale）。
    3. 将缩放后的FP16梯度复制到FP32主参数上。
    4. 在FP32主参数上将梯度除以缩放因子，还原为真实梯度。
    5. 使用优化器在FP32主参数上进行权重更新。
    6. 将更新后的FP32主参数复制回FP16模型参数。
    7. 动态调整损失缩放因子：如果梯度出现NaN或inf（上溢），则减小缩放因子；
       如果一段时间内没有上溢，则增大缩放因子，以利用更宽的动态范围。
    """
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth # 对数损失缩放因子的增长率

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params # 默认情况下，主参数就是模型参数
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale # 对数形式的损失缩放因子

        if self.use_fp16:
            # 如果启用FP16，则进行混合精度设置
            # 1. 对模型参数进行分组
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            # 2. 创建FP32主参数
            self.master_params = make_master_params(self.param_groups_and_shapes)
            # 3. 将模型转换为FP16
            self.model.convert_to_fp16()

    def zero_grad(self):
        # 清空模型参数的梯度。
        # 在每次计算梯度之前调用。
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        # 执行反向传播。
        if self.use_fp16:
            # 如果使用FP16，应用损失缩放。
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            # 否则，执行常规的反向传播。
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        # 执行优化器步骤。
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer):
        # FP16模式下的优化步骤。
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        # 1. 将FP16模型梯度复制到FP32主参数
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        # 2. 计算梯度的范数，用于检查是否溢出（NaN/inf）
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            # 如果梯度溢出：
            # a. 减小损失缩放因子，以在下次迭代时使用更小的缩放。
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            # b. 清空主参数梯度，跳过本次参数更新。
            zero_master_grads(self.master_params)
            return False # 表示本次优化失败

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        # 3. 将主参数的梯度除以缩放因子，还原为真实梯度。
        for p in self.master_params:
            p.grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        
        # 4. 在FP32主参数上执行优化器步骤。
        opt.step()
        
        # 5. 清空主参数梯度，为下一次迭代做准备。
        zero_master_grads(self.master_params)
        
        # 6. 将更新后的FP32主参数复制回FP16模型参数。
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        
        # 7. 如果本次更新成功，可以略微增大损失缩放因子，以探索更大的梯度范围。
        self.lg_loss_scale += self.fp16_scale_growth
        return True # 表示本次优化成功

    def _optimize_normal(self, opt: th.optim.Optimizer):
        # 正常（FP32）模式下的优化步骤。
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        # 计算所有主参数的范数和梯度的范数。
        # 这对于监控训练过程和检测梯度爆炸/消失非常有用。
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        # 返回梯度的真实范数（除以缩放因子）和参数的范数。
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        # 辅助方法，用于将主参数保存到 state_dict。
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        # 辅助方法，用于从 state_dict 加载主参数。
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    # 检查一个值是否是无穷大（inf）或非数值（NaN）。
    # 在混合精度训练中，这用于检测梯度上溢。
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
