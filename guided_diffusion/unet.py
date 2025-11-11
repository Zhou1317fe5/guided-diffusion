from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
  

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    任何一个 forward() 方法接受时间步嵌入（timestep embeddings）作为第二个参数的模块的基类。
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        给定时间步嵌入 `emb`，将模块应用于 `x`。
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    一个序列化的模块，它能将时间步嵌入（timestep embeddings）传递给支持该输入的子模块。
    在处理一系列网络层时，能够识别出哪些层需要额外的时间步信息（`emb`），并将这个信息只传递给它们；而对于那些不需要时间步信息的普通层，则像往常一样只传递主要数据（`x`）
    
    input:一个是主要的图像数据 x，另一个是代表当前时间步的嵌入向量 emb
    
    
    """

    def forward(self, x, emb):
        # 遍历所有层
        for layer in self:
            # 如果层是 TimestepBlock 的实例子类，说明它需要时间步嵌入
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            # 否则，正常前向传播
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    一个带有可选卷积的上采样层。

    :param channels: 输入和输出的通道数。
    :param use_conv: 一个布尔值，决定是否应用卷积。
    :param dims: 决定信号是一维、二维还是三维。如果是三维，则在内两维进行上采样。
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        # 如果使用卷积，则定义一个3x3卷积层
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # 根据维度选择不同的插值方法
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            # 使用最近邻插值将特征图尺寸放大两倍
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        # 如果使用卷积，则在插值后应用卷积
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    一个带有可选卷积的下采样层。

    :param channels: 输入和输出的通道数。
    :param use_conv: 一个布尔值，决定是否应用卷积。
    :param dims: 决定信号是一维、二维还是三维。如果是三维，则在内两维进行下采样。
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        # 如果使用卷积，则使用步长为2的卷积进行下采样
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            # 否则，使用平均池化进行下采样
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    一个残差块，可以选择性地改变通道数。

    :param channels: 输入通道数。
    :param emb_channels: 时间步嵌入的通道数。
    :param dropout: dropout 比率。
    :param out_channels: 如果指定，则为输出通道数。
    :param use_conv: 如果为 True 且指定了 out_channels，则在跳跃连接中使用空间卷积
                     而不是 1x1 卷积来改变通道数。
    :param dims: 决定信号是一维、二维还是三维。
    :param use_checkpoint: 如果为 True，则在此模块上使用梯度检查点。
    :param up: 如果为 True，则此块用于上采样。
    :param down: 如果为 True，则此块用于下采样。
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # 输入层：归一化 -> SiLU激活 -> 3x3卷积
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        # 上采样或下采样层
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # 时间步嵌入层，用于将时间信息注入到残差块中
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                # 如果使用 scale-shift-norm，输出通道数是两倍（scale 和 shift）
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        # 输出层：归一化 -> SiLU激活 -> Dropout -> 3x3卷积（初始化为零）
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # 跳跃连接，用于匹配输入和输出的通道数
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        将残差块应用于张量，并以时间步嵌入为条件。

        :param x: 一个 [N x C x ...] 的特征张量。
        :param emb: 一个 [N x emb_channels] 的时间步嵌入张量。
        :return: 一个 [N x C x ...] 的输出张量。
        """
        # 使用梯度检查点来节省内存
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # 如果是上采样或下采样块
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        # 处理时间步嵌入
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        # 将时间步嵌入注入到网络中
        if self.use_scale_shift_norm:
            # 使用 FiLM-like 的调节机制 (scale and shift)
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            # 直接将嵌入加到特征上
            h = h + emb_out
            h = self.out_layers(h)
        
        # 返回跳跃连接和主路径输出的和
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    一个注意力块，允许空间位置相互关注。

    最初从这里移植，但已适配到 N 维情况。
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # 先分割 qkv，再分割头
            self.attention = QKVAttention(self.num_heads)
        else:
            # 先分割头，再分割 qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        # 使用梯度检查点来节省内存
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        # 计算 Q, K, V
        qkv = self.qkv(self.norm(x))
        # 应用注意力机制
        h = self.attention(qkv)
        # 投影回原始维度
        h = self.proj_out(h)
        # 添加残差连接
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    一个用于 `thop` 包的计数器，用于计算注意力操作中的运算量。    
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # 我们执行两次具有相同操作数的矩阵乘法。
    # 第一次计算权重矩阵，第二次计算值向量的组合。
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    一个执行 QKV 注意力的模块。匹配旧版的 QKVAttention + 输入/输出头的塑形。
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        应用 QKV 注意力。

        :param qkv: 一个 [N x (H * 3 * C) x T] 的 Qs, Ks, 和 Vs 张量。
        :return: 注意力之后的一个 [N x (H * C) x T] 张量。
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # 使用 f16 时比后除更稳定
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    一个以不同顺序执行 QKV 注意力和分割的模块。
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        应用 QKV 注意力。

        :param qkv: 一个 [N x (3 * H * C) x T] 的 Qs, Ks, 和 Vs 张量。
        :return: 注意力之后的一个 [N x (H * C) x T] 张量。
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # 使用 f16 时比后除更稳定
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    一个带有注意力和时间步嵌入的完整 UNet 模型。

    这是一个经典的编码器-解码器（Encoder-Decoder）结构。
    编码器部分（input_blocks）通过一系列残差块和下采样层逐步提取特征，减小空间维度。
    解码器部分（output_blocks）通过一系列残差块和上采样层逐步恢复空间维度，并生成最终输出。
    编码器和解码器之间通过跳跃连接（skip connections）相连，将编码器中对应层级的特征
    （存储在 hs 列表中）与解码器中的特征进行拼接，以保留高分辨率的细节信息。
    模型的中间部分（middle_block）在编码器和解码器之间应用了残差块和注意力块。

    :param in_channels: 输入张量的通道数。
    :param model_channels: 模型的基础通道数。
    :param out_channels: 输出张量的通道数。
    :param num_res_blocks: 每个下采样/上采样级别中的残差块数量。
    :param attention_resolutions: 一个集合，指定在哪些下采样率下应用注意力机制。
                                  例如，如果包含 4，则在 4x 下采样时将使用注意力。
    :param dropout: dropout 概率。
    :param channel_mult: UNet 每个级别的通道数乘数。
    :param conv_resample: 如果为 True，则使用可学习的卷积进行上采样和下采样。
    :param dims: 决定信号是一维、二维还是三维。
    :param num_classes: 如果指定（作为整数），则此模型将是类别条件的，
                        具有 `num_classes` 个类别。
    :param use_checkpoint: 使用梯度检查点以减少内存使用。
    :param num_heads: 每个注意力层中的注意力头数。
    :param num_heads_channels: 如果指定，则忽略 num_heads，而是为每个注意力头使用固定的通道宽度。
    :param num_heads_upsample: 与 num_heads 配合使用，为上采样设置不同的头数。已弃用。
    :param use_scale_shift_norm: 使用类似 FiLM 的条件机制。
    :param resblock_updown: 使用残差块进行上/下采样。
    :param use_new_attention_order: 使用不同的注意力模式以可能提高效率。
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # 时间步嵌入网络
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 类别嵌入网络（如果模型是类别条件的）
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # --- 编码器（Encoder）部分 ---
        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1 # ds表示缩放尺寸

        # 遍历每个分辨率级别
        for level, mult in enumerate(channel_mult):
            # 在每个级别内添加指定数量的残差块
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                # 如果当前分辨率需要注意力机制，则添加一个注意力块
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            # 如果不是最后一级，则添加一个下采样层
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # --- 中间（Middle）部分 ---
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # --- 解码器（Decoder）部分 ---
        self.output_blocks = nn.ModuleList([])
        # 反向遍历每个分辨率级别
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # 在每个级别内添加指定数量的残差块（+1 是因为要处理来自跳跃连接的特征）
            for i in range(num_res_blocks + 1):
                # 从编码器中取出对应层的通道数
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,  # 输入通道数是当前通道数 + 跳跃连接的通道数
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                # 如果当前分辨率需要注意力机制，则添加一个注意力块
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                # 如果不是第一级，并且是该级别的最后一个块，则添加一个上采样层
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # --- 输出层 ---
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        将模型的主体部分转换为 float16。
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        将模型的主体部分转换为 float32。
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        将模型应用于一个输入批次。

        数据流：
        1. 计算时间步嵌入 `emb` 和类别嵌入 `y`（如果提供）。
        2. 输入 `x` 依次通过 `input_blocks`（编码器）。
        3. 在每个 `input_block` 之后，其输出 `h` 被存储在 `hs` 列表中，用于后续的跳跃连接。
        4. 编码器的最终输出 `h` 通过 `middle_block`。
        5. `h` 依次通过 `output_blocks`（解码器）。
        6. 在每个 `output_block` 之前，`h` 与从 `hs` 列表中弹出的对应编码器层的输出进行拼接（`th.cat`）。
           这就是跳跃连接的实现。
        7. 最终，解码器的输出通过 `out` 层得到最终结果。

        :param x: 一个 [N x C x ...] 的输入张量。
        :param timesteps: 一个一维的时间步批次。
        :param y: 一个 [N] 的标签张量，如果模型是类别条件的。
        :return: 一个 [N x C x ...] 的输出张量。
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        # 1. 计算时间步嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # 如果是类别条件的，添加类别嵌入
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        # 2. 编码器部分
        for module in self.input_blocks:
            h = module(h, emb)
            # 3. 存储跳跃连接的特征
            hs.append(h)
        # 4. 中间部分
        h = self.middle_block(h, emb)
        # 5. 解码器部分
        for module in self.output_blocks:
            # 6. 拼接跳跃连接的特征
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        # 7. 输出层
        return self.out(h)


class SuperResModel(UNetModel):
    """
    一个执行超分辨率的 UNetModel。

    期望一个额外的关键字参数 `low_res` 来以低分辨率图像为条件。
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        # 输入通道数是原始图像通道数 + 低分辨率图像通道数
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        # 将低分辨率图像上采样到与输入 x 相同的尺寸
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        # 将输入 x 和上采样后的低分辨率图像在通道维度上拼接
        x = th.cat([x, upsampled], dim=1)
        # 调用父类的 forward 方法
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    一个只有一半的 UNet 模型（即编码器），带有注意力和时间步嵌入。

    用法参见 UNetModel。
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        将模型的主体部分转换为 float16。
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        将模型的主体部分转换为 float32。
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        将模型应用于一个输入批次。

        :param x: 一个 [N x C x ...] 的输入张量。
        :param timesteps: 一个一维的时间步批次。
        :return: 一个 [N x K] 的输出张量。
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)
