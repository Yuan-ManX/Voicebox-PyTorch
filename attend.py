from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce


# 定义 FlashAttentionConfig 命名元组，用于配置 FlashAttention 的参数
"""
FlashAttentionConfig 命名元组用于配置 FlashAttention 的参数。

参数说明:
    enable_flash (bool): 是否启用 Flash 优化。
    enable_math (bool): 是否启用数学优化。
    enable_mem_efficient (bool): 是否启用内存高效优化。
"""
FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    """
    检查一个值是否存在（即不为 None）。

    参数:
        val (Optional[Any]): 要检查的值。

    返回:
        bool: 如果值存在（即不为 None），则返回 True；否则返回 False。
    """
    return val is not None


def default(val, d):
    """
    如果值存在（即不为 None），则返回该值；否则返回默认值。

    参数:
        val (Optional[Any]): 要检查的值。
        d (Any): 默认值。

    返回:
        Any: 如果值存在，则返回该值；否则返回默认值。
    """
    return val if exists(val) else d


def once(fn):
    """
    装饰器，确保被装饰的函数只被调用一次。

    参数:
        fn (Callable): 被装饰的函数。

    返回:
        Callable: 包装后的函数，确保只调用一次。

    示例:
        @once
        def initialize():
            print("初始化")

        initialize()  # 输出: 初始化
        initialize()  # 无输出
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 使用 once 装饰器创建一个只打印一次的 print 函数
print_once = once(print)


class Attend(nn.Module):
    """
    Attend 类实现了一个自注意力机制模块，支持 Flash Attention 和传统的自注意力计算。
    该模块可以根据硬件条件选择最合适的注意力计算方式，以提高计算效率和性能。

    参数说明:
        dropout (float, 可选): Dropout 失活概率，默认为0。
        flash (bool, 可选): 是否启用 Flash Attention，默认为 False。
        scale (float, 可选): 注意力机制的缩放因子，默认为 None。
    """
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        # Dropout 层
        self.attn_dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = scale

        # 是否启用 Flash Attention
        self.flash = flash
        # 检查是否启用了 Flash Attention 并且 PyTorch 版本是否支持
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu
        # 确定在 CUDA 和 CPU 上高效注意力配置

        # CPU 上启用所有优化
        self.cpu_config = FlashAttentionConfig(True, True, True)
        # 初始化 CUDA 配置
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            # 如果没有 CUDA 或不启用 Flash Attention，则返回
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # 如果是 A100 GPU，打印提示
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            # 仅启用 Flash 优化
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            # 如果不是 A100 GPU，打印提示
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            # 启用数学和内存高效优化
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask = None):
        """
        使用 Flash Attention 计算自注意力。

        参数:
            q (Tensor): 查询张量，形状为 (B, H, Q, D)。
            k (Tensor): 键张量，形状为 (B, H, K, D)。
            v (Tensor): 值张量，形状为 (B, H, K, D)。
            mask (Tensor, 可选): 注意力掩码，形状为 (B, Q)。

        返回:
            Tensor: 注意力输出，形状为 (B, H, Q, D)。
        """
        # 获取张量形状和设备信息
        _, heads, q_len, dim_head, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # if scale is given, divide by the default scale that sdpa uses
        # 如果提供了缩放因子，则根据默认的缩放因子进行调整
        if exists(self.scale):
            q = q * (self.scale / (dim_head ** -0.5))

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        # 如果存在掩码，则将其扩展到兼容的形状
        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        # 确定注意力配置
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        前向传播方法，执行自注意力计算。

        参数:
            q (Tensor): 查询张量，形状为 (B, H, N, D)。
            k (Tensor): 键张量，形状为 (B, H, N, D)。
            v (Tensor): 值张量，形状为 (B, H, N, D)。
            mask (Tensor, 可选): 注意力掩码，形状为 (B, N)。

        返回:
            Tensor: 注意力输出，形状为 (B, H, N, D)。

        einstein表示法:
            b - 批量大小
            h - 注意力头数
            n, i, j - 序列长度（基础序列长度，源，目标）
            d - 特征维度
        """
        # 获取序列长度和设备信息
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # 计算缩放因子
        scale = default(self.scale, q.shape[-1] ** -0.5)

        if exists(mask) and mask.ndim != 4:
            # 调整掩码形状
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            # 使用 Flash Attention
            return self.flash_attn(q, k, v, mask = mask)

        # similarity
        # 计算相似度矩阵
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask
        # 应用键掩码
        if exists(mask):
            # 应用掩码
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention
        # 计算注意力权重
        # 计算 softmax
        attn = sim.softmax(dim=-1)
        # 应用 Dropout
        attn = self.attn_dropout(attn)

        # aggregate values
        # 计算最终输出
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
