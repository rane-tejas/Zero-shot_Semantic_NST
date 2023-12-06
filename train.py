import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader

from utils import *
from models.decoder import Decoder
from models.vgg_encoder import ATA_Encoder
from models.AdaAttN import AdaAttN, Transformer
from datasets import PhraseCutDataset
from losses import LossFunctions

torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainStyleTransfer():

    def __init__(self):

        # TODO: pass args asn hyperparams

        self.encoder = ATA_Encoder().to(DEVICE)
        self.ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64).to(DEVICE)
        self.transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64).to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.loss = LossFunctions()
    
    def train(self):
            
        dataset = PhraseCutDataset("./dataset/PhraseCut_mod")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        for p in self.encoder.parameters():
            p.requires_grad = False

        parameters = []
        parameters.extend(list(self.ada_attn_3.parameters()))
        parameters.extend(list(self.transformer.parameters()))
        parameters.extend(list(self.decoder.parameters()))
        optimizer = torch.optim.Adam(parameters, lr=0.001)

        _avg_loss = 0.0

        for epoch in range(1):
            for batch in dataloader:

                content_images = batch[0].to(DEVICE)
                style_images = batch[1].to(DEVICE)

                optimizer.zero_grad()

                content_features = self.encoder(content_images)
                style_features = self.encoder(style_images)
                c_adain_feat_3 = self.ada_attn_3(content_features[2], style_features[2], get_key(content_features, 2), get_key(style_features, 2))
                cs = self.transformer(content_features[3], style_features[3], content_features[4], style_features[4],
                                    get_key(content_features, 3), get_key(style_features, 3),
                                    get_key(content_features, 4), get_key(style_features, 4))
                cs = self.decoder(cs, c_adain_feat_3)

                enc_cs = self.encoder(cs)


                content_loss = self.loss.content_loss(enc_cs, content_features)
                style_loss = self.loss.style_loss(enc_cs, content_features, style_features)

                loss = content_loss + style_loss

                loss.backward()
                optimizer.step()

                _avg_loss += loss.item()

        print("Average Loss: ", _avg_loss/100)


if __name__=="__main__":
    
    # TODO:
    # 1. Setup training args (hyperparams)
    # 2. 

    train_instance = TrainStyleTransfer()
    train_instance.train()