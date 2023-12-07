import os
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

    def __init__(self, checkpoint_path, log_path, lr=0.001, weight_decay=0.0):

        self._logger = Logger(log_path)

        self.lr = lr
        self.weight_decay = weight_decay
        self.checkpoint_path = checkpoint_path
        self.ckpt_path = log_path + '/ckpt'
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.parameters = []
        self.optimizer = None

        self.encoder = ATA_Encoder(self.checkpoint_path).to(DEVICE)
        self.ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64, checkpoint_path=self.checkpoint_path).to(DEVICE)
        self.transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64, checkpoint_path=self.checkpoint_path).to(DEVICE)
        self.decoder = Decoder(self.checkpoint_path).to(DEVICE)
        self.loss = LossFunctions()

        self._build_models()

    def _build_models(self):

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.parameters.extend(list(self.ada_attn_3.parameters()))
        self.parameters.extend(list(self.transformer.parameters()))
        self.parameters.extend(list(self.decoder.parameters()))

    def _train_epoch(self, content_images, style_images):

        self.optimizer.zero_grad()

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
        self.optimizer.step()

        return loss.item()

    def _val_epoch(self, content_images, style_images):

        self.encoder.eval()
        self.transformer.eval()
        self.decoder.eval()
        self.ada_attn_3.eval()

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

        return loss.item()

    def train(self, dataset_path, num_epochs, batch_size):

        train_dataset = PhraseCutDataset(dataset_path+'/train')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = PhraseCutDataset(dataset_path+'/val')
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        # _avg_loss = 0.0
        _train_batches = len(train_dataloader)
        _val_batches = len(val_dataloader)
        max_loss = 1e9

        self._logger.log(tag='args', lr=self.lr, weight_decay=self.weight_decay,
                         dataset_path=dataset_path, checkpoint_path=self.checkpoint_path, num_epochs=num_epochs, batch_size=batch_size)

        print('Starting training...')

        for epoch in range(num_epochs):
            _loss = 0.0
            _avg_loss = 0.0

            for batch in train_dataloader:

                content_images = batch[0].to(DEVICE)
                style_images = batch[1].to(DEVICE)

                _loss = self._train_epoch(content_images, style_images)
                _avg_loss += _loss

            _avg_loss = _avg_loss / _train_batches
            print(f'Epoch {epoch}, Average Training Loss: {_avg_loss}')
            self._logger.log(tag='train', epoch=epoch, loss=_avg_loss)

            _loss = 0.0
            _avg_loss = 0.0

            for batch in val_dataloader:

                content_images = batch[0].to(DEVICE)
                style_images = batch[1].to(DEVICE)

                _loss = self._val_epoch(content_images, style_images)
                _avg_loss += _loss

            _avg_loss = _avg_loss / _val_batches
            print(f'Epoch {epoch}, Average Validation Loss: {_avg_loss}')
            self._logger.log(tag='val', epoch=epoch, loss=_avg_loss)

            if max_loss > _avg_loss:
                print('Saving best model')
                max_loss = _avg_loss
                torch.save(self.ada_attn_3.state_dict(), self.log_path+'/adaattn.pth')
                torch.save(self.transformer.state_dict(), self.log_path+'/transformer.pth')
                torch.save(self.decoder.state_dict(), self.log_path+'/decoder.pth')

        print('Training complete')
        self._logger.log(tag='plot')


if __name__=="__main__":

    args = infer_args()

    train_instance = TrainStyleTransfer(args.checkpoint_path, args.log_path, args.lr, args.weight_decay)
    train_instance.train(args.dataset_path, args.num_epochs, args.batch_size)