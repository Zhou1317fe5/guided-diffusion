# 从 abc 模块导入 ABC（抽象基类）和 abstractmethod（抽象方法），用于定义抽象类。
from abc import ABC, abstractmethod

# 导入 numpy 用于高效的数值计算。
import numpy as np
# 导入 torch 用于构建和训练神经网络。
import torch as th
# 导入 torch.distributed 用于分布式训练，支持多 GPU 或多机器环境。
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    根据给定的名称创建一个 ScheduleSampler（时间步采样器）。
    这是一个工厂函数，用于从预定义的采样器库中选择和实例化一个采样器。

    :param name: 采样器的名称 (例如, "uniform", "loss-second-moment")。
    :param diffusion: 扩散模型对象，采样器将为这个对象进行采样。
    :return: 一个 ScheduleSampler 的实例。
    """
    if name == "uniform":
        # 如果名称是 "uniform"，返回一个均匀采样器。
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        # 如果名称是 "loss-second-moment"，返回一个基于损失二阶矩的重采样器。
        return LossSecondMomentResampler(diffusion)
    else:
        # 如果名称未知，则抛出未实现错误。
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    一个关于扩散过程中时间步（timesteps）的分布，旨在减少训练目标的方差。
    通过对时间步进行重要性采样，可以更有效地训练模型，让模型更关注那些损失较大或较不稳定的时间步。

    默认情况下，采样器执行无偏重要性采样（unbiased importance sampling），
    在这种情况下，训练目标的均值保持不变。
    然而，子类可以重写 sample() 方法来改变重采样项的权重，从而实现对训练目标的实际调整。
    """

    @abstractmethod
    def weights(self):
        """
        获取一个 numpy 数组形式的权重，每个扩散步骤对应一个权重。
        这个方法必须在子类中实现。

        权重不需要归一化，但必须是正数。
        """

    def sample(self, batch_size, device):
        """
        为一个批次（batch）进行时间步的重要性采样。

        :param batch_size: 批次大小，即需要采样的时间步数量。
        :param device: 用于存储张量的 torch 设备 (例如, 'cpu' 或 'cuda')。
        :return: 一个元组 (timesteps, weights):
                 - timesteps: 一个张量，包含采样到的时间步索引。
                 - weights: 一个张量，用于缩放相应损失的权重。这些权重是根据重要性采样计算得出的，
                            用于修正由于非均匀采样而引入的偏差，以确保训练目标的期望值不变。
        """
        # 获取原始权重
        w = self.weights()
        # 将权重归一化为概率分布
        p = w / np.sum(w)
        # 根据概率分布 p 从 [0, len(p)-1] 中随机选择 batch_size 个索引
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        # 将 numpy 数组转换为 torch 张量，并移动到指定设备
        indices = th.from_numpy(indices_np).long().to(device)
        # 计算重要性采样的权重，公式为 1 / (N * P(i))，其中 N 是总时间步数，P(i) 是采样到时间步 i 的概率。
        # 这个权重用于在计算损失时对每个样本进行加权，以纠正非均匀采样带来的偏差。
        weights_np = 1 / (len(p) * p[indices_np])
        # 将权重数组转换为 torch 张量
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    均匀采样器。
    这是最简单的采样策略，它为每个时间步分配完全相同的权重。
    这意味着在训练过程中，每个时间步被选中的概率是相等的。
    """
    def __init__(self, diffusion):
        self.diffusion = diffusion
        # 创建一个长度为总时间步数的全 1 数组作为权重。
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        # 返回全 1 的权重数组。
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    一个能感知损失的采样器（抽象基类）。
    这种采样器会根据模型在不同时间步上的损失来动态调整采样权重。
    其目的是让模型更多地关注那些难以学习（即损失较高）的时间步。
    """
    def update_with_local_losses(self, local_ts, local_losses):
        """
        使用来自一个模型的局部损失来更新重采样权重。

        在分布式训练的每个 rank（进程）上调用此方法，传入一批时间步和对应的损失。
        此方法将执行同步操作，以确保所有 rank 保持完全相同的重采样权重。

        :param local_ts: 一个整数张量，表示时间步。
        :param local_losses: 一个一维张量，表示对应的损失。
        """
        # 获取分布式环境中的 world size (总进程数)
        world_size = dist.get_world_size()
        # 准备一个列表，用于收集所有 rank 的批次大小
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(world_size)
        ]
        # 使用 all_gather 从所有 rank 收集批次大小
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # 将批次大小列表转换为 Python int 列表，并找到最大批次大小
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        # 准备用于 all_gather 的填充张量列表
        # 因为 all_gather 要求所有进程的张量形状相同，所以需要将数据填充到最大批次大小
        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        # 从所有 rank 收集时间步和损失数据
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # 将收集到的数据扁平化为两个列表：timesteps 和 losses
        # 这里会去除填充的部分
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        # 使用收集到的所有损失来更新权重
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        使用来自所有进程的全部损失来更新重采样权重。

        子类应重写此方法，以实现具体的权重更新逻辑。

        此方法直接更新权重，无需在工作进程之间进行同步。
        它由 update_with_local_losses 从所有 rank 以相同的参数调用。
        因此，它的行为应该是确定性的，以保持所有工作进程之间的状态一致。

        :param ts: 一个整数列表，表示时间步。
        :param losses: 一个浮点数列表，每个时间步对应一个损失。
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    基于损失二阶矩的重采样器。
    这种策略根据每个时间步损失的二阶矩（即 E[L^2]）来计算采样权重。
    损失的二阶矩可以衡量损失的大小和其不确定性（方差）。
    给二阶矩较大的时间步更高的权重，可以让模型更关注那些损失大或不稳定的时间步。

    与均匀采样相比，这种方法可以加速收敛并可能提高最终性能。
    """
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        """
        初始化方法。
        :param diffusion: 扩散模型对象。
        :param history_per_term: 每个时间步要保留的损失历史记录的数量。
        :param uniform_prob: 在最终的权重中混入的均匀分布的概率。
                             这可以确保即使某些时间步的计算出的权重非常低，它们仍然有被采样的机会，
                             增加了采样的多样性，防止模型完全忽略某些时间步。
        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        # 初始化一个二维数组，用于存储每个时间步的损失历史。
        # 形状为 (总时间步数, 每个时间步的历史记录数)
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        # 初始化一个一维数组，用于记录每个时间步已收集的损失数量。
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """
        计算采样权重。
        """
        # 检查是否所有时间步的损失历史都已“预热”（即收集了足够数量的损失记录）。
        if not self._warmed_up():
            # 如果没有预热完成，则返回均匀权重，表现得像一个 UniformSampler。
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        
        # 计算每个时间步损失历史的平方均值的平方根，即损失的二阶矩的估计。
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        # 归一化权重
        weights /= np.sum(weights)
        # 乘以 (1 - uniform_prob)
        weights *= 1 - self.uniform_prob
        # 加上一小部分的均匀分布权重，以确保所有时间步都有被采样的机会。
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        用新的损失数据更新损失历史记录。
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        检查是否所有时间步的损失历史记录都已填满。
        只有当每个时间步都收集了 history_per_term 个损失值后，才认为采样器已“预热”。
        """
        return (self._loss_counts == self.history_per_term).all()
