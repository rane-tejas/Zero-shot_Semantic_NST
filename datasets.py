import os
import cv2
import torch
import numpy as np

from natsort import natsorted
from torchvision import transforms
from torch.utils.data import Dataset

np.random.seed(42)
torch.manual_seed(42)

class PhraseCutDataset(Dataset):

    def __init__(self, data_dir):
        super(PhraseCutDataset, self).__init__()

        self.data_dir = data_dir
        self.resize = transforms.Resize(512, antialias=None)
        self.content_image_list = natsorted(os.listdir(os.path.join(self.data_dir, "content")))
        self.style_image_list = natsorted(os.listdir(os.path.join(self.data_dir, "style")))

    def __len__(self):
        return len(self.content_image_list)

    def __getitem__(self,index):

        content_image_path = os.path.join(self.data_dir, "content", self.content_image_list[index])
        style_image_path = os.path.join(self.data_dir, "style", np.random.choice(self.style_image_list))

        # print(content_image_path)
        # print(style_image_path)

        content_img = torch.Tensor(cv2.imread(content_image_path).transpose(2, 0, 1))
        style_img = torch.Tensor(cv2.imread(style_image_path).transpose(2, 0, 1))
        content_shape = content_img.shape[1:]

        content_img = self.resize(content_img)
        style_img = self.resize(style_img)

        return content_img, style_img, content_shape

