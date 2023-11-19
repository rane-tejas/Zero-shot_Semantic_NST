import cv2
from torch.utils.data import Dataset,DataLoader
import torch
import glob
import random

class StyleTransferDataset(Dataset):

    def __init__(self,img_folder_path):
        super(StyleTransferDataset,self).__init__()
        self.img_folder_path = img_folder_path
        self.content_images_list = glob.glob(self.img_folder_path+"/content/*.jpg")
        self.style_images_list = glob.glob(self.img_folder_path+"/style/*.jpg")

    def __len__(self):
        return len(glob.glob(self.img_folder_path+"/content/*.jpg"))

    def __getitem__(self,index):

        ##TODO: Check RGB vs BGR

        content_img = cv2.imread(self.content_images_list[index])
        style_img = cv2.imread(random.choice(self.style_images_list))

        content_img = cv2.resize(content_img, (512, 512))
        style_img = cv2.resize(style_img,(512,512))




        return content_img, style_img

