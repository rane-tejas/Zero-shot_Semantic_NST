import os
import cv2
import torch
import numpy as np

from natsort import natsorted
from torchvision import transforms
from torch.utils.data import Dataset

from utils import *

np.random.seed(42)
torch.manual_seed(42)

class PhraseCutDataset(Dataset):

    def __init__(self, data_dir):
        super(PhraseCutDataset, self).__init__()

        self.data_dir = data_dir
        self.resize = transforms.Resize((512, 512), antialias=None)
        #TODO: Add change listdir to glob
        self.content_image_list = natsorted(os.listdir(os.path.join(self.data_dir, "content")))
        self.style_image_list = natsorted(os.listdir(os.path.join(self.data_dir, "style")))

    def __len__(self):
        return len(self.content_image_list)

    def __getitem__(self, index):

        content_image_path = os.path.join(self.data_dir, "content", self.content_image_list[index])
        style_image_path = os.path.join(self.data_dir, "style", np.random.choice(self.style_image_list))

        _content_img = cv2.imread(content_image_path)
        _style_img = cv2.imread(style_image_path)

        content_img = img_to_tensor(cv2.cvtColor(padding(_content_img, 32), cv2.COLOR_BGR2RGB)).squeeze(0)
        style_img = img_to_tensor(cv2.cvtColor(padding(_style_img, 32), cv2.COLOR_BGR2RGB)).squeeze(0)
        content_shape = _content_img.shape

        content_img = self.resize(content_img)
        style_img = self.resize(style_img)

        return content_img, style_img, content_shape

