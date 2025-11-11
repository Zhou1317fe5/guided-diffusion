import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    为一个数据集创建一个生成器，用于生成 (images, kwargs) 对。

    每个 images 是一个 NCHW 格式的浮点数张量，kwargs 字典包含零个或多个键，
    每个键都映射到一个批处理后的张量。
    kwargs 字典可用于类别标签，此时键为 "y"，值为类别标签的整数张量。

    :param data_dir: 数据集目录。
    :param batch_size: 每个返回对的批处理大小。
    :param image_size: 图像被调整到的大小。
    :param class_cond: 如果为 True，则在返回的字典中包含一个 "y" 键用于类别标签。
                       如果类别不可用且此项为 True，将引发异常。
    :param deterministic: 如果为 True，则按确定性顺序生成结果。
    :param random_crop: 如果为 True，则对图像进行随机裁剪以进行数据增强。
    :param random_flip: 如果为 True，则对图像进行随机翻转以进行数据增强。
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    # 对data_dir所有图片进行遍历
    all_files = _list_image_files_recursively(data_dir) # 递归地列出目录中所有的图片文件
    
    classes = None
    if class_cond:
        # 假设类别是文件名的第一部分，以下划线分隔。
        # 例如：cat_01.jpg, dog_01.jpg
        class_names = [bf.basename(path).split("_")[0] for path in all_files] # ['cat', 'dog']
        # 将类别名映射到整数索引
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))} # 对class_names进行排序 {'bird': 0, 'cat': 1, 'dog': 2}
        classes = [sorted_classes[x] for x in class_names] # class_names 是 ['cat', 'dog', 'cat']，则 classes 会是 [1, 2, 1]。
    
    # 创建 ImageDataset 实例
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        # 在分布式训练中，shard 用于指定当前进程处理的数据分片
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    
    # 根据是否需要确定性顺序，创建不同的 DataLoader
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    # 使用 while True 和 yield from 创建一个无限循环的数据加载器
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    递归地遍历一个目录，返回所的图像文件路径列表。
    支持的格式包括: jpg, jpeg, png, gif。
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            # 如果是子目录，则递归调用
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    """
    一个用于加载和处理图像的 PyTorch 数据集类。
    它支持分布式训练中的数据分片和基本的数据增强（随机裁剪和翻转）。
    """
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        # 根据分片（shard）信息，只加载当前进程负责的图像
        # 例如，如果有 4 个进程，进程 0 会加载第 0, 4, 8, ... 个图像
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        """返回当前分片中的图像数量。"""
        return len(self.local_images)

    def __getitem__(self, idx):
        """
        根据索引加载、处理并返回单个图像及其元数据。
        """
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        # 确保图像是 RGB 格式
        pil_image = pil_image.convert("RGB")

        # 根据设置进行随机裁剪或中心裁剪
        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        # 根据设置和随机数决定是否进行水平翻转
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1] # 水平翻转

        # 将像素值从 [0, 255] 范围归一化到 [-1, 1] 范围
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        # 将 HWC 格式的数组转置为 CHW 格式，以符合 PyTorch 的要求
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    """
    对 PIL 图像进行中心裁剪。
    为了提高下采样质量，首先使用 BOX 滤波器将图像尺寸逐步缩小到目标尺寸的两倍以内，
    然后再使用 BICUBIC 滤波器进行精确缩放，最后执行中心裁剪。
    """
    # 为了提高下采样质量，我们手动进行逐步下采样。
    # 当图像的最小边长大于等于目标尺寸的两倍时，使用 BOX 滤波器将其尺寸减半。
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # 计算缩放比例，使图像的最小边等于 image_size
    scale = image_size / min(*pil_image.size)
    # 使用 BICUBIC 滤波器进行高质量缩放
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # 将 PIL 图像转换为 numpy 数组
    arr = np.array(pil_image)
    # 计算裁剪的起始坐标，以确保裁剪区域位于中心
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """
    对 PIL 图像进行随机裁剪。
    首先将图像随机缩放到一个尺寸，然后从中随机裁剪出一个 image_size x image_size 的区域。
    """
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    # 在指定范围内随机选择一个较小的维度尺寸
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # 与中心裁剪类似，为了提高下采样质量，先进行逐步下采样
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # 计算缩放比例，使图像的最小边等于随机选择的 smaller_dim_size
    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # 将 PIL 图像转换为 numpy 数组
    arr = np.array(pil_image)
    # 随机选择裁剪的起始坐标
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
