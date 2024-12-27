from pathlib import Path
from functools import wraps
from einops import rearrange
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchaudio


def exists(val):
    """
    检查一个值是否存在（即不为 None）。

    参数:
        val (Any): 要检查的值。

    返回:
        bool: 如果值存在（即不为 None），则返回 True；否则返回 False。
    """
    return val is not None


def cast_tuple(val, length = 1):
    """
    将输入值转换为元组。如果输入已经是元组，则保持不变；否则，将其重复指定次数以形成元组。

    参数:
        val (Any): 输入值，可以是单个值或元组。
        length (int, 可选): 如果输入不是元组，则重复的次数，默认为1。

    返回:
        Tuple[Any, ...]: 转换后的元组。
    """
    return val if isinstance(val, tuple) else ((val,) * length)


class AudioDataset(Dataset):
    """
    AudioDataset 类继承自 torch.utils.data.Dataset，用于加载音频数据。
    该数据集从指定的文件夹中加载所有指定扩展名的音频文件，并提供迭代访问每个音频样本的方法。

    参数说明:
        folder (str 或 Path): 包含音频文件的文件夹路径。
        audio_extension (str, 可选): 音频文件的扩展名，默认为 ".flac"。
    """
    @beartype
    def __init__(
        self,
        folder,
        audio_extension = ".flac"
    ):
        super().__init__()
        # 将文件夹路径转换为 Path 对象
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        # 存储音频文件扩展名
        self.audio_extension = audio_extension

        # 使用 glob 查找所有匹配音频扩展名的文件
        files = list(path.glob(f'**/*{audio_extension}'))
        assert len(files) > 0, 'no files found'
        
        # 存储文件列表
        self.files = files

    def __len__(self):
        """
        获取数据集的长度，即音频文件的数量。

        返回:
            int: 数据集的长度。
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        获取指定索引的音频样本。

        参数:
            idx (int): 样本的索引。

        返回:
            Tensor: 加载的音频数据，形状为 (C, T)。
        """
        file = self.files[idx]
        # 加载音频文件
        wave, _ = torchaudio.load(file)
        # 重塑张量形状，去除批量维度
        wave = rearrange(wave, '1 ... -> ...')

        return wave


def collate_one_or_multiple_tensors(fn):
    """
    装饰器，用于处理单个或多个张量的批处理函数。
    如果数据是单个张量，则直接应用批处理函数；
    如果数据是多个张量，则分别对每个张量应用批处理函数。

    参数:
        fn (Callable): 批处理函数，用于处理单个或多个张量。

    返回:
        Callable: 包装后的批处理函数。
    """
    @wraps(fn)
    def inner(data):
        # 检查数据是否为单个张量
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            # 如果是单个张量，则直接应用批处理函数
            data = fn(data)
            # 返回一个元组
            return (data,)

        outputs = []
        # 对数据进行解包并遍历
        for datum in zip(*data):
            # 检查数据是否为字符串元组
            if is_bearable(datum, Tuple[str, ...]):
                # 如果是字符串元组，则转换为列表
                output = list(datum)
            else:
                # 否则，应用批处理函数
                output = fn(datum)
            # 将处理后的数据添加到输出列表中
            outputs.append(output)
        # 返回处理后的数据元组
        return tuple(outputs)
    
    # 返回包装后的函数
    return inner


@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    """
    对数据进行批处理，裁剪所有张量到最短长度，然后堆叠成一个批次张量。

    参数:
        data (Tuple[torch.Tensor, ...]): 输入数据，一个包含多个张量的元组。

    返回:
        torch.Tensor: 裁剪并堆叠后的批次张量。
    """
    # 找到所有张量的最短长度
    min_len = min(*[datum.shape[0] for datum in data])
    # 裁剪每个张量到最短长度
    data = [datum[:min_len] for datum in data]
    # 将裁剪后的张量堆叠成一个批次张量
    return torch.stack(data)


@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    """
    对数据进行批处理，对所有张量进行填充，使其长度相同，然后堆叠成一个批次张量。

    参数:
        data (Tuple[torch.Tensor, ...]): 输入数据，一个包含多个张量的元组。

    返回:
        torch.Tensor: 填充并堆叠后的批次张量。
    """
    # 对张量进行填充，使其长度相同，并堆叠成批次张量
    return pad_sequence(data, batch_first = True)


def get_dataloader(ds, pad_to_longest = True, **kwargs):
    """
    获取数据加载器，根据是否需要填充到最长长度选择合适的批处理函数。

    参数:
        ds (Dataset): 数据集。
        pad_to_longest (bool, 可选): 是否将所有样本填充到最长长度，默认为 True。
        **kwargs: 其他传递给 DataLoader 的关键字参数。

    返回:
        DataLoader: 配置好的数据加载器。
    """
    # 选择批处理函数
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    # 返回配置好的数据加载器
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
