import os
import torch
import numpy as np
from torch.utils import data
from processing import ImgProcessing


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=192, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root

        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root))) # 每张图片路径

        self.mode = mode

        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        img_name = self.image_paths[index]
        img_c = ImgProcessing(img_name) # 打开图片 (图片对象)
        img_c.inten_normal() # 归一化
        img_c.z_score() # 标准化
        img = img_c.get_array()  # (图片矩阵)

        img_name = img_name[:-3] + 'png' # 数据后缀为.jpg 标签后缀为.png 替换后缀以加载后面的mask
        mask = load_mask(img_name)
        return torch.from_numpy(np.expand_dims(img, 0)).float(), torch.from_numpy(np.expand_dims(mask, 0)).float()

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def load_img(path):
    img = ImgProcessing(path)  # 读入图片
    return img.get_array()


def load_mask(path):
    mask = ImgProcessing(path.replace('images', 'masks'))
    mask_array = mask.get_array()
    return (mask_array / 255).astype(np.uint8)  # 0 1
    # 原mask为8位深图像 黑为0 白为255


def get_loader(image_path, batch_size, num_workers=2, mode='train', shuffle=False):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader
