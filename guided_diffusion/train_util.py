import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
# 初始的对数损失缩放因子。对于ImageNet实验，这是一个很好的默认值。
# 我们发现 lg_loss_scale 在训练的前1000步内会迅速攀升到20-21。
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    """
    一个封装了整个训练循环的类，负责模型训练的核心流程。
    """
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        """
        初始化训练循环。

        :param model: 要训练的扩散模型。
        :param diffusion: 扩散过程的辅助类，用于计算损失等。
        :param data: 数据加载器，提供训练数据批次。
        :param batch_size: 每个全局批次的大小。
        :param microbatch: 每个微批次的大小。用于梯度累积，如果为-1或0，则不使用梯度累积。
        :param lr: 学习率。
        :param ema_rate: Exponential Moving Average (EMA) 的衰减率，可以是一个浮点数或逗号分隔的字符串。
        :param log_interval: 记录日志的步数间隔。
        :param save_interval: 保存模型检查点的步数间隔。
        :param resume_checkpoint: 用于恢复训练的检查点文件路径。
        :param use_fp16: 是否使用16位浮点数（混合精度）进行训练。
        :param fp16_scale_growth: FP16动态损失缩放的增长因子。
        :param schedule_sampler: 时间步采样器，用于在训练中为每个样本选择t。
        :param weight_decay: AdamW优化器的权重衰减。
        :param lr_anneal_steps: 学习率线性退火的总步数。如果为0，则不进行退火。
        """
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        # 如果 microbatch 小于等于0，则将其设置为 batch_size，意味着不进行梯度累积。
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        # 如果未提供 schedule_sampler，则使用均匀采样器。
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        # 全局批次大小 = 单个GPU的批次大小 * GPU数量
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        # 加载并同步模型参数，如果提供了 resume_checkpoint，则会从检查点加载。
        self._load_and_sync_parameters()
        
        # 初始化混合精度训练器
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        # 初始化 AdamW 优化器，作用于 FP32 的主参数。
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            # 如果从检查点恢复，则加载优化器状态和 EMA 参数。
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            # 否则，初始化 EMA 参数为当前主参数的深拷贝。
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        # 设置分布式数据并行 (DDP)
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        """
        从检查点加载模型参数，并在所有 DDP 进程之间同步。
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            # 从文件名解析恢复的步数
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                # 在主进程上加载模型状态字典
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        
        # 在所有进程之间同步模型参数，确保初始状态一致。
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        """
        为给定的 EMA 速率加载 EMA 参数。
        """
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                # 将加载的状态字典转换为 FP32 主参数
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        # 同步 EMA 参数
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        """
        从检查点加载优化器状态。
        """
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        """
        主训练循环。
        """
        # 循环直到达到学习率退火的总步数
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # 从数据加载器获取下一个批次
            batch, cond = next(self.data)
            # 执行一个训练步骤
            self.run_step(batch, cond)
            # 按间隔记录日志
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            # 按间隔保存模型
            if self.step % self.save_interval == 0:
                self.save()
                # 在集成测试中，运行有限的时间后退出。
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # 如果最后一个检查点没有被保存，则在训练结束时保存。
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        """
        执行单个训练步骤。
        """
        # 执行前向传播和反向传播，计算梯度。
        self.forward_backward(batch, cond)
        # 使用优化器更新模型参数。
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            # 如果参数被更新，则更新 EMA 参数。
            self._update_ema()
        # 对学习率进行退火。
        self._anneal_lr()
        # 记录当前步骤的信息。
        self.log_step()

    def forward_backward(self, batch, cond):
        """
        执行前向和反向传播，实现梯度累积。
        """
        # 清零梯度
        self.mp_trainer.zero_grad()
        # 遍历批次中的所有微批次
        for i in range(0, batch.shape[0], self.microbatch):
            # 提取当前微批次数据和条件
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            # 检查是否是最后一个微批次
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # 采样时间步 t 和对应的权重
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # 创建一个偏函数来计算损失
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            # 如果是最后一个微批次或者不使用 DDP，则正常计算损失并同步梯度。
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                # 在非最后一个微批次时，使用 no_sync 上下文管理器来避免不必要的梯度同步，
                # 从而实现梯度累积。
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                # 如果使用 LossAwareSampler，用当前批次的损失来更新采样器。
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            # 计算加权损失
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            # 使用混合精度训练器进行反向传播
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        """
        更新所有 EMA 参数。
        """
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        """
        对学习率进行线性退火。
        """
        if not self.lr_anneal_steps:
            return
        # 计算训练完成的比例
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        # 计算新的学习率
        lr = self.lr * (1 - frac_done)
        # 更新优化器中的学习率
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """
        记录当前训练步骤的统计信息。
        """
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """
        保存模型检查点，包括模型参数、EMA 参数和优化器状态。
        """
        def save_checkpoint(rate, params):
            """辅助函数，用于保存单个检查点。"""
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    # 保存主模型
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    # 保存 EMA 模型
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        # 保存主模型参数
        save_checkpoint(0, self.mp_trainer.master_params)
        # 保存所有 EMA 参数
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # 保存优化器状态
        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # 设置屏障，确保所有进程都完成了保存操作再继续。
        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    从文件名中解析步数。
    例如，解析 "path/to/modelNNNNNN.pt" 格式的文件名，其中 NNNNNN 是检查点的步数。
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    """
    获取用于保存检查点的日志目录。
    可以修改此函数以将检查点保存到 blob 存储或其他外部驱动器。
    """
    return logger.get_dir()


def find_resume_checkpoint():
    """
    查找要恢复的检查点。
    在实际的基础设施中，您可能希望重写此函数以自动发现 blob 存储上的最新检查点。
    """
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    根据主检查点、步数和 EMA 速率查找对应的 EMA 检查点。
    """
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    记录损失字典中的各项损失。
    """
    for key, values in losses.items():
        # 记录每个损失项的均值
        logger.logkv_mean(key, values.mean().item())
        # 记录损失的四分位数
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
