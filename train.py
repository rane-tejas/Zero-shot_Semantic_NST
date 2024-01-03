import os
import cv2
import time
import torch
import numpy as np

from torch.utils.data import DataLoader

from utils import *
from models.decoder import Decoder
from models.vgg_encoder import Encoder
from models.AdaAttN import AdaAttN, Transformer
from datasets import PhraseCutDataset
from losses import LossFunctions

torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainStyleTransfer():
    """ Class to train the style transfer model

    Args:
        checkpoint_path (str): path to the checkpoint
        log_path (str): path to the log
        lr (float): learning rate
        weight_decay (float): weight decay
        msg (str): message to be logged
    """

    def __init__(self, checkpoint_path, log_path, lr=0.001, weight_decay=0.0, msg="", lc=1.0, lg=1.0, ll=1.0):

        self._logger = Logger(log_path)

        self.lr = lr
        self.lg = lg
        self.ll = ll
        self.lc = lc
        self.msg = msg
        self.weight_decay = weight_decay
        self.checkpoint_path = checkpoint_path
        self.ckpt_path = log_path + '/ckpt'
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.parameters = []
        self.optimizer = None

        self.encoder = Encoder(self.checkpoint_path).to(DEVICE)
        self.ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64, checkpoint_path=self.checkpoint_path).to(DEVICE)
        self.transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64, checkpoint_path=self.checkpoint_path).to(DEVICE)
        self.decoder = Decoder(self.checkpoint_path).to(DEVICE)
        self.loss = LossFunctions(lambda_content=self.lc, lambda_global=self.lg, lambda_local=self.ll)

        self._build_models()

    def _build_models(self):
        """ Build the models and freeze the encoder """

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.parameters.extend(list(self.ada_attn_3.parameters()))
        self.parameters.extend(list(self.transformer.parameters()))
        self.parameters.extend(list(self.decoder.parameters()))

    def _train_epoch(self, content_images, style_images):
        """ Train the model for one epoch.

        Args:
            content_images (torch.Tensor): content images
            style_images (torch.Tensor): style images

        Returns:
            float: loss
        """

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
        """ Validate the model for one epoch.

        Args:
            content_images (torch.Tensor): content images
            style_images (torch.Tensor): style images

        Returns:
            float: loss
        """

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

    def _infer(self, content_images=None, style_images=None):
        """ Infer the model for one epoch.

        Args:
            content_images (torch.Tensor): content images
            style_images (torch.Tensor): style images

        Returns:
            cs (torch.Tensor): stylized images
        """

        self.encoder.eval()
        self.transformer.eval()
        self.decoder.eval()
        self.ada_attn_3.eval()

        if content_images == None:
            _content_img = cv2.imread("data/content/c1.jpg")
            _style_img = cv2.imread("data/style/vg_starry_night.jpg")
            content_img = resize_img(_content_img, 512, keep_ratio=False)
            style_img = resize_img(_style_img, 512, keep_ratio=False)
            content_images = img_to_tensor(cv2.cvtColor(padding(content_img, 32), cv2.COLOR_BGR2RGB)).to(DEVICE)
            style_images = img_to_tensor(cv2.cvtColor(padding(style_img, 32), cv2.COLOR_BGR2RGB)).to(DEVICE)

        content_features = self.encoder(content_images)
        style_features = self.encoder(style_images)
        c_adain_feat_3 = self.ada_attn_3(content_features[2], style_features[2], get_key(content_features, 2), get_key(style_features, 2))
        cs = self.transformer(content_features[3], style_features[3], content_features[4], style_features[4],
                              get_key(content_features, 3), get_key(style_features, 3),
                              get_key(content_features, 4), get_key(style_features, 4))
        cs = self.decoder(cs, c_adain_feat_3)
        cs = tensor_to_img(cs)
        cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)

        return cs

    def train(self, dataset_path, num_epochs, batch_size):
        """ Train the model.

        Args:
            dataset_path (str): path to the dataset
            num_epochs (int): number of epochs
            batch_size (int): batch size
        """

        train_dataset = PhraseCutDataset(dataset_path+'/train')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = PhraseCutDataset(dataset_path+'/val')
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        _train_batches = len(train_dataloader)
        _val_batches = len(val_dataloader)
        max_loss = 1e9

        self._logger.log(tag='args', lr=self.lr, weight_decay=self.weight_decay,
                         dataset_path=dataset_path, checkpoint_path=self.checkpoint_path, num_epochs=num_epochs, batch_size=batch_size,
                         total_train_images=(_train_batches*batch_size), total_val_images=(_val_batches*batch_size), message=self.msg,
                         lambda_content=self.lc, lambda_global=self.lg, lambda_local=self.ll)

        print('Starting training...')

        for epoch in range(num_epochs):
            _loss = 0.0
            _avg_loss = 0.0
            _start = time.time()

            for batch in train_dataloader:

                content_images = batch[0].to(DEVICE)
                style_images = batch[1].to(DEVICE)

                _loss = self._train_epoch(content_images, style_images)
                _avg_loss += _loss

            _avg_loss = _avg_loss / _train_batches
            print(f'Epoch {epoch}, Average Training Loss: {_avg_loss}')
            self._logger.log(tag='train', epoch=epoch, loss=_avg_loss, time=(time.time()-_start))

            _loss = 0.0
            _avg_loss = 0.0
            _start = time.time()

            for batch in val_dataloader:

                content_images = batch[0].to(DEVICE)
                style_images = batch[1].to(DEVICE)

                _loss = self._val_epoch(content_images, style_images)
                _avg_loss += _loss

            _avg_loss = _avg_loss / _val_batches
            print(f'Epoch {epoch}, Average Validation Loss: {_avg_loss}')
            self._logger.log(tag='val', epoch=epoch, loss=_avg_loss, time=(time.time()-_start))

            if max_loss > _avg_loss:
                print('Saving best model')
                self._logger.log(tag='model', loss=_avg_loss)

                cs = self._infer(content_images, style_images)
                self._logger.draw(epoch, cs)

                max_loss = _avg_loss
                torch.save(self.encoder.state_dict(), self.ckpt_path+'/encoder.pth')
                torch.save(self.ada_attn_3.state_dict(), self.ckpt_path+'/adaattn.pth')
                torch.save(self.transformer.state_dict(), self.ckpt_path+'/transformer.pth')
                torch.save(self.decoder.state_dict(), self.ckpt_path+'/decoder.pth')

            self._logger.log(tag='plot')
            if epoch % 5 == 0:
                cs = self._infer()
                self._logger.draw(epoch, cs)

        print('Training complete')


if __name__=="__main__":

    args = train_args()

    train_instance = TrainStyleTransfer(args.checkpoint_path, args.log_dir+args.log_name, args.lr, args.weight_decay, args.msg,
                                        args.lc, args.lg, args.ll)
    train_instance.train(args.dataset_path, args.num_epochs, args.batch_size)