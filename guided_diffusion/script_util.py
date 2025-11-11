import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, EncoderUNetModel

# 预定义的类别数量，通常用于分类任务
NUM_CLASSES = 1000


def diffusion_defaults():
    """
    为图像和分类器训练提供默认的扩散模型参数。
    这些是扩散过程的核心配置，控制噪声如何添加以及模型如何学习去噪。
    """
    return dict(
        learn_sigma=False,  # 是否学习扩散过程中的方差(sigma)
        diffusion_steps=1000,  # 扩散过程的总步数
        noise_schedule="linear",  # 噪声调度策略，可以是 "linear", "cosine" 等
        timestep_respacing="",  # 时间步重采样策略，用于加速采样过程
        use_kl=False,  # 是否使用KL散度作为损失的一部分，通常用于学习方差
        predict_xstart=False,  # 模型是预测去噪后的图像 (x_start) 还是预测噪声
        rescale_timesteps=False,  # 是否重新缩放时间步，以改善模型在不同时间步的性能
        rescale_learned_sigmas=False,  # 是否重新缩放学习到的方差
    )


def classifier_defaults():
    """
    为分类器模型提供默认参数。
    这些参数定义了分类器网络的结构和行为。
    """
    return dict(
        image_size=64,  # 输入图像的尺寸
        classifier_use_fp16=False,  # 是否使用16位浮点数进行训练，以加速并减少内存使用
        classifier_width=128,  # 分类器模型中的通道数（宽度）
        classifier_depth=2,  # 分类器模型中残差块的数量（深度）
        classifier_attention_resolutions="32,16,8",  # 在哪些分辨率下使用注意力机制
        classifier_use_scale_shift_norm=True,  # 是否在归一化层中使用尺度和偏移参数
        classifier_resblock_updown=True,  # 是否在残差块中使用上采样/下采样
        classifier_pool="attention",  # 池化策略，"attention" 表示使用注意力池化
    )


def model_and_diffusion_defaults():
    """
    为图像生成的U-Net模型和扩散过程提供合并的默认参数。
    这个函数整合了模型结构和扩散过程的配置。
    """
    res = dict(
        image_size=64,  # 图像尺寸
        num_channels=128,  # U-Net模型的基础通道数
        num_res_blocks=2,  # 每个分辨率下的残差块数量
        num_heads=4,  # 注意力机制中的头数
        num_heads_upsample=-1,  # 上采样注意力头数，-1表示与num_heads相同
        num_head_channels=-1,  # 每个注意力头的通道数，-1表示自动计算
        attention_resolutions="16,8",  # 在哪些分辨率下使用注意力机制
        channel_mult="",  # 通道数乘数，用于控制不同分辨率下的通道数
        dropout=0.0,  # Dropout比率
        class_cond=False,  # 是否使用类别条件
        use_checkpoint=False,  # 是否使用检查点来节省内存
        use_scale_shift_norm=True,  # 是否在归一化层中使用尺度和偏移
        resblock_updown=False,  # 是否在残差块中使用上采样/下采样
        use_fp16=False,  # 是否使用16位浮点数
        use_new_attention_order=False,  # 是否使用新的注意力计算顺序
    )
    res.update(diffusion_defaults())  # 将扩散模型的默认参数合并进来
    return res


def classifier_and_diffusion_defaults():
    """
    为分类器模型和扩散过程提供合并的默认参数。
    """
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    image_size,  # 图像尺寸
    class_cond,  # 是否使用类别条件进行引导，生成模型有条件的还是无条件的
    learn_sigma,  # 是否学习扩散过程的方差
    num_channels,  # 模型的基础通道数
    num_res_blocks,  # 每个分辨率下的残差块数量
    channel_mult,  # 不同分辨率下的通道数乘数
    num_heads,  # 注意力机制的头数
    num_head_channels,  # 每个注意力头的通道数
    num_heads_upsample,  # 上采样中的注意力头数
    attention_resolutions,  # 在哪些res_blocks上要进行attention
    dropout,  # Dropout比率
    diffusion_steps,  # 扩散过程的总步数
    noise_schedule,  # 噪声调度策略
    timestep_respacing,  # 时间步重采样策略
    use_kl,  # 是否使用KL散度作为损失
    predict_xstart,  # 模型是预测x_start还是预测噪声
    rescale_timesteps,  # 是否重新缩放时间步
    rescale_learned_sigmas,  # 是否重新缩放学习到的方差
    use_checkpoint,  # 是否使用检查点以节省内存
    use_scale_shift_norm,  # 是否使用尺度和偏移归一化
    resblock_updown,  # 是否在残差块中使用上/下采样
    use_fp16,  # 是否使用16位浮点数精度
    use_new_attention_order,  # 是否使用新的注意力计算顺序
):
    """
    根据传入的参数创建U-Net模型和高斯扩散过程。

    :param image_size: 图像尺寸。
    :param class_cond: 是否使用类别条件。生成模型有条件的还是无条件的
    :param learn_sigma: 是否学习方差。
    ... (其他参数与 model_and_diffusion_defaults 中的键对应)
    :return: 一个U-Net模型实例和一个高斯扩散实例。
    """
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    """
    创建一个U-Net模型。

    :param image_size: 图像尺寸。
    :param num_channels: 基础通道数。
    :param num_res_blocks: 残差块数量。
    :param channel_mult: 通道数乘数。如果为空，则根据图像大小自动设置。
    ... (其他参数定义了U-Net的结构和行为)
    :return: 一个UNetModel实例。
    """
    # 根据图像大小自动设置通道数乘数
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        # 从字符串解析通道数乘数
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    # 解析需要应用注意力机制的分辨率
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # 实例化并返回U-Net模型
    return UNetModel(
        image_size=image_size,
        in_channels=3,  # 输入通道数，通常为RGB图像的3
        model_channels=num_channels,  # 模型的基础通道数
        out_channels=(3 if not learn_sigma else 6),  # 输出通道数，如果学习sigma则为6，否则为3
        num_res_blocks=num_res_blocks,  # 残差块数量
        attention_resolutions=tuple(attention_ds),  # 应用注意力的分辨率
        dropout=dropout,
        channel_mult=channel_mult,  # 通道数乘数
        num_classes=(NUM_CLASSES if class_cond else None),  # 如果有类别条件，则传入类别数
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_classifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    """
    根据传入的参数创建分类器模型和高斯扩散过程。

    :param image_size: 图像尺寸。
    ... (其他参数与 classifier_and_diffusion_defaults 中的键对应)
    :return: 一个分类器模型实例和一个高斯扩散实例。
    """
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    """
    创建一个分类器模型。

    :param image_size: 图像尺寸。
    ... (其他参数定义了分类器的结构和行为)
    :return: 一个EncoderUNetModel实例作为分类器。
    """
    # 根据图像大小自动设置通道数乘数
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    # 解析需要应用注意力机制的分辨率
    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # 实例化并返回分类器模型
    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,  # 输出通道数，对应类别数
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


def sr_model_and_diffusion_defaults():
    """
    为超分辨率（Super-Resolution）模型提供默认参数。
    它基于通用的模型和扩散默认值，并添加了特定于超分任务的参数。
    """
    res = model_and_diffusion_defaults()
    res["large_size"] = 256  # 目标大图像的尺寸
    res["small_size"] = 64   # 输入小图像的尺寸
    # 过滤掉不适用于超分模型的参数
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
):
    """
    创建用于超分辨率任务的模型和扩散过程。

    :param large_size: 目标大图像的尺寸。
    :param small_size: 输入小图像的尺寸。
    ... (其他参数与 sr_model_and_diffusion_defaults 中的键对应)
    :return: 一个超分模型实例和一个高斯扩散实例。
    """
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
):
    """
    创建一个超分辨率模型。

    :param large_size: 目标大图像的尺寸。
    :param small_size: 输入小图像的尺寸。
    ... (其他参数定义了超分模型的结构)
    :return: 一个SuperResModel实例。
    """
    _ = small_size  # 避免 "unused variable" 警告

    # 根据目标图像大小设置通道数乘数
    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    # 解析需要应用注意力机制的分辨率
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    # 实例化并返回超分模型
    return SuperResModel(
        image_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    创建高斯扩散过程实例。

    :param steps: 扩散总步数。
    :param learn_sigma: 是否学习方差。
    :param sigma_small: 如果不学习方差，是使用大的固定方差还是小的。
    :param noise_schedule: 噪声调度策略。
    :param use_kl: 是否使用KL散度损失。
    :param predict_xstart: 模型是预测x_start还是噪声。
    :param rescale_timesteps: 是否重新缩放时间步。
    :param rescale_learned_sigmas: 是否重新缩放学习到的方差。
    :param timestep_respacing: 时间步重采样策略。采样的步数
    :return: 一个SpacedDiffusion实例。
    """
    # 获取beta调度表，得到一个加噪方案
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    # 根据参数选择损失类型
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    # 如果没有指定重采样的步数，则默认使用所有步
    if not timestep_respacing:
        timestep_respacing = [steps]
    # 实例化并返回SpacedDiffusion
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    将字典中的键值对作为参数添加到argparse.ArgumentParser实例中。

    :param parser: argparse.ArgumentParser的实例。
    :param default_dict: 包含默认参数的字典。
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool  # 对布尔值使用特殊处理函数
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    从argparse解析后的结果中提取指定的键，并将其转换为字典。

    :param args: argparse.parse_args() 的返回结果。
    :param keys: 需要提取的键的列表。
    :return: 一个包含指定键值对的字典。键从keys中得到，值是从args中得到
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    将字符串转换为布尔值，用于argparse。
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
