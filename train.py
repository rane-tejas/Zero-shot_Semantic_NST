import cv2
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

from utils import *
from models.decoder import Decoder
from models.vgg_encoder import Encoder
from models.AdaAttN import AdaAttN, Transformer
from dataloader import StyleTransferDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StyleTransferModel(nn.Module):

    def __init__(self):
        super(StyleTransferModel,self).__init__()
        self.encoder = Encoder()
        self.transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64)
        self.decoder = Decoder()
    
    def forward(self,content_img,style_img):

        content_embedding = self.encoder(content_img)
        style_embedding = self.encoder(style_img)

        transformer_output = self.transformer(content_embedding,style_embedding)

        styled_image = self.decoder(transformer_output)





if __name__=="__main__":

    model = StyleTransferModel()
    model.train()

    model.to(DEVICE)

    batch_size=2

    dataset = StyleTransferDataset("./dataset/PhraseCut_mod")
    dataloader = DataLoader(dataset,batch_size,shuffle=True)

    for batch in dataloader:

        content_images = batch[0]
        style_images = batch[1]

        print(content_images.shape)
        print(style_images.shape)

        ## Model Pass
        ## Encoder 
        ## Compute loss
        ## Loss backward
        ## Optimizer step

        
        break





