import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    从原始的扩散过程中创建一系列要使用的时间步。

    这个函数根据给定的 `section_counts`（每个部分要采样的步数），
    从原始的总时间步中，生成一个新的、更稀疏的时间步序列。
    这对于加速采样过程至关重要，因为它允许我们跳过一些计算步骤。

    例如，如果总共有 300 个时间步，`section_counts` 是 [10, 15, 20]，
    那么原始的 300 步会被分成三部分，每部分 100 步。
    - 前 100 步 (0-99) 中，会均匀采样 10 个时间步。
    - 中间 100 步 (100-199) 中，会均匀采样 15 个时间步。
    - 最后 100 步 (200-299) 中，会均匀采样 20 个时间步。

    如果 `section_counts` 是一个以 "ddim" 开头的字符串（例如 "ddim25"），
    则会使用 DDIM 论文中提出的固定步长采样方法，此时只允许一个部分。

    :param num_timesteps: 原始扩散过程中的总步数。
    :param section_counts: 可以是一个列表或逗号分隔的字符串，表示每个部分要采样的步数。
                           特殊情况是 "ddimN"，其中 N 是步数，表示使用 DDIM 的采样策略。
    :return: 一个集合，包含从原始过程中选择出的时间步。
    """
    if isinstance(section_counts, str):
        # --- 处理 "ddim" 模式 ---
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :]) # 获取N
            # DDIM 采样策略要求一个固定的步长（stride）。
            # 为了找到一个合适的步长 `i`，使得采样步数恰好等于 `desired_count`，
            # 这里通过循环遍历所有可能的步长值（从 1 到 num_timesteps-1）。
            # `range(0, num_timesteps, i)` 会生成一个等差序列，其长度就是采样步数。
            # 当找到一个步长 `i` 使得序列长度等于 `desired_count` 时，就返回这个时间步集合。
            for i in range(1, num_timesteps): # 创建一个从1到num_timesteps-1的循环，变量i代表候选的步长值。
                if len(range(0, num_timesteps, i)) == desired_count:  # len(range(0, num_timesteps, i)): 计算使用步长i能够产生的采样点数量。if len(...) == desired_count: 检查当前步长i产生的采样点数量是否正好等于用户期望的步数。
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {desired_count} steps with an integer stride"
            )
        # 如果是普通字符串，则按逗号分割并转换为整数列表
        section_counts = [int(x) for x in section_counts.split(",")]

    # --- 处理分段采样模式 ---
    # size_per: 计算每个分段的基础大小。例如，1000 步分为 4 段，每段基础大小为 250。
    size_per = num_timesteps // len(section_counts)
    # extra: 计算不能整除时的余数。例如，1001 步分为 4 段，余数为 1。
    # 这个余数会均匀分配给前面的分段，使得每个分段的大小最多只差 1。
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        # 计算当前分段的实际大小。如果 `i < extra`，说明这个分段需要分配一个额外的步数。
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        # `frac_stride`: 计算在当前分段内采样的步长（可以是浮点数）。
        # 这是为了在 `size` 这么多的步数里，均匀地选出 `section_count` 个点。
        # 公式 (size - 1) / (section_count - 1) 确保了起点和终点都被包含在内。
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            # 通过累加 `frac_stride` 并四舍五入，得到在当前分段内最接近的整数时间步。
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
        
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    一个可以跳过基础扩散过程中某些步骤的扩散过程。

    这个类继承自 `GaussianDiffusion`，并允许使用一个稀疏的时间步集合 `use_timesteps`
    来创建一个新的、步数更少的等效扩散过程。它通过重新计算 beta 和 alpha_cumprod
    来实现这一点，从而在保持扩散过程特性的同时减少计算量。

    :param use_timesteps: 一个序列或集合，包含从原始扩散过程中要保留的时间步。
    :param kwargs: 用于创建基础 `GaussianDiffusion` 实例的参数。
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []  # 新时间步索引 -> 原始时间步索引的映射
        self.original_num_steps = len(kwargs["betas"]) # 原始加噪步数

        # --- 重新计算 betas 和 alphas_cumprod ---
        # 1. 创建一个基础的扩散过程实例，这个实例包含了原始的、完整的 betas 和 alphas_cumprod。
        #    `base_diffusion` 的作用是提供计算新 betas 所需的原始 alpha_cumprod 值。
        base_diffusion = GaussianDiffusion(**kwargs)
        
        # 2. `last_alpha_cumprod` 用于存储上一个被保留的时间步的 alpha_cumprod 值。
        #    在循环中，它代表 alpha_cumprod_{t-1}。初始值为 1.0，对应 alpha_cumprod_{-1}。
        last_alpha_cumprod = 1.0
        new_betas = []
        
        # 3. 遍历原始的所有时间步（从 0 到 T-1）。
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            # 4. 只处理在 `use_timesteps` 中指定要保留的时间步。
            if i in self.use_timesteps:
                # 5. 基于保留的 alpha_cumprod 重新计算新的 beta 值。
                #    公式为: new_beta_t = 1 - alpha_cumprod_t / alpha_cumprod_{t-1}
                #    这里的 alpha_cumprod_t 是当前保留步的 alpha_cumprod，
                #    而 last_alpha_cumprod 是上一个保留步的 alpha_cumprod。
                #    这个公式推导自 alpha_cumprod_t = (1 - beta_t) * alpha_cumprod_{t-1}。
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                
                # 6. 更新 `last_alpha_cumprod` 为当前步的 alpha_cumprod，供下一次计算使用。
                last_alpha_cumprod = alpha_cumprod
                
                # 7. 记录原始时间步 `i`。`timestep_map` 用于将新的、连续的时间步索引
                #    （例如 0, 1, 2, ...）映射回原始的、稀疏的时间步（例如 0, 2, 5, ...）。
                self.timestep_map.append(i)
        
        # 8. 使用新计算出的 `new_betas` 替换掉原始的 `betas`。
        kwargs["betas"] = np.array(new_betas)
        
        # 9. 调用父类的构造函数 `GaussianDiffusion.__init__`。
        #    此时，父类会使用我们提供的、更短的 `new_betas` 数组来初始化所有扩散过程相关的参数
        #    （如 alphas, alphas_cumprod 等），从而创建一个步数更少但效果等价的扩散过程。
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):  
        # 在调用父类方法前，包装模型以处理时间步映射
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):  
        # 在调用父类方法前，包装模型以处理时间步映射
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        # 在调用父类方法前，包装条件函数以处理时间步映射
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        # 在调用父类方法前，包装条件函数以处理时间步映射
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        """
        包装模型或条件函数，使其能够处理稀疏时间步到原始时间步的映射。
        """
        # 这行检查是为了防止模型被重复包装。
        # 如果一个模型已经是 `_WrappedModel` 的实例，说明它已经被包装过了，
        # 无需再次包装，直接返回即可。这可以避免不必要的开销和潜在的错误。
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # 在 SpacedDiffusion 中，时间步的缩放操作已经由 _WrappedModel 处理，
        # 因此这里直接返回原始的、未缩放的时间步 `t` 即可。
        # 父类 `GaussianDiffusion` 中的 `_scale_timesteps` 会将时间步缩放到 [0, 1000)，
        # 但在这里我们希望在包装器中进行更精确的映射和缩放。
        return t


class _WrappedModel:
    """
    一个内部帮助类，用于包装模型，将稀疏的时间步索引映射回原始的时间步索引。
    """
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        """
        在调用底层模型之前，将稀疏时间步 `ts` 转换为原始时间步。
        :param x: 模型输入张量。
        :param ts: 一个批量的、稀疏的时间步张量 (例如，张量中包含 [0, 1, 2, ..., 49] 等值)。
        :param kwargs: 传递给模型的其他关键字参数。
        """
        # `self.timestep_map` 是一个列表，例如 [0, 2, 5, 8, ...]。
        # `map_tensor` 将这个列表转换为一个 PyTorch 张量，以便进行高效的索引操作。
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        
        # `ts` 是一个包含新时间步索引的张量。例如，如果一个批次包含时间步 [3, 0, 4]，
        # `map_tensor[ts]` 会利用张量索引一次性地、并行地查找到对应的原始时间步。
        # 比如，`map_tensor[[3, 0, 4]]` 会返回 `[map_tensor[3], map_tensor[0], map_tensor[4]]`，
        # 即 `[8, 0, 12]`（假设 timestep_map[3]=8, timestep_map[0]=0, timestep_map[4]=12）。
        # 这样就实现了从稀疏时间步到原始时间步的批量映射。
        new_ts = map_tensor[ts]
        
        # `rescale_timesteps` 是一个布尔标志，用于控制是否将时间步缩放到 [0, 1000) 的范围。
        # 某些预训练模型可能期望输入的时间步是在一个固定的范围（如 0-1000）内。
        # 如果为 True，这里会将映射后的原始时间步（例如 8, 0, 12）按比例缩放。
        # 例如，如果原始总步数是 4000，时间步 8 会被缩放为 8 * (1000.0 / 4000) = 2.0。
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            
        # 使用映射并可能缩放后的时间步 `new_ts` 来调用原始模型。
        return self.model(x, new_ts, **kwargs)
