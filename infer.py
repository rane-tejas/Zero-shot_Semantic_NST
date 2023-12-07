import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader

from utils import *
from datasets import PhraseCutDataset
from models.decoder import Decoder
from models.vgg_encoder import ATA_Encoder
from models.AdaAttN import AdaAttN, Transformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InferStyleTransfer:

    def __init__(self, checkpoint_path):

        self.seg_mask = None
        self.style_img = None
        self.content_img = None
        self.content_shape = None

        self.image_encoder = ATA_Encoder(checkpoint_path).to(DEVICE)
        self.decoder = Decoder(checkpoint_path).to(DEVICE)
        self.ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64, checkpoint_path=checkpoint_path).to(DEVICE)
        self.transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64, checkpoint_path=checkpoint_path).to(DEVICE)

    def build_models(self):

        # self.image_encoder.load_state_dict(torch.load(checkpoint_path+'/vgg_normalised.pth'))
        # self.transformer.load_state_dict(torch.load(checkpoint_path+'/latest_net_transformer.pth'))
        # self.decoder.load_state_dict(torch.load(checkpoint_path+'/latest_net_decoder.pth'))
        # self.ada_attn_3.load_state_dict(torch.load(checkpoint_path+'/latest_net_adaattn_3.pth'))

        self.image_encoder.eval()
        self.transformer.eval()
        self.decoder.eval()
        self.ada_attn_3.eval()

        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.ada_attn_3.parameters():
            p.requires_grad = False

    def load_images(self, content_path, mask_path, style_path, resize=True, keep_ratio=True):

        self.content_img = cv2.imread(content_path)
        self.content_shape = self.content_img.shape[:2]
        self.style_img = cv2.imread(style_path)

        if mask_path:
            self.seg_mask = cv2.imread(mask_path)//255

        if resize:
            self.content_img = resize_img(self.content_img, 512, keep_ratio)
            self.style_img = resize_img(self.style_img, 512, keep_ratio)

    def load_images_from_dataset(self, data_path, mask_path, resize=True, keep_ratio=True):

        test_dataloader = DataLoader(PhraseCutDataset(data_path), batch_size=1, shuffle=True)
        for batch in test_dataloader:
            self.content_img, self.style_img, self.content_shape = batch
            # import ipdb; ipdb.set_trace()
            self.content_img = torch.permute(self.content_img.squeeze(), (1, 2, 0)).numpy()
            self.style_img = torch.permute(self.style_img.squeeze(), (1, 2, 0)).numpy()
            break

        if mask_path:
            self.seg_mask = cv2.imread(mask_path)//255

        # if resize:
        #     self.content_img = resize_img(self.content_img, 512, keep_ratio)
        #     self.style_img = resize_img(self.style_img, 512, keep_ratio)


    def run(self, content_path, mask_path, style_path, resize=True, keep_ratio=True):

        self.load_images(content_path, mask_path, style_path, resize, keep_ratio)
        # self.load_images_from_dataset(content_path, mask_path)
        self.build_models()

        with torch.no_grad():
            style = img_to_tensor(cv2.cvtColor(padding(self.style_img, 32), cv2.COLOR_BGR2RGB)).to(DEVICE)
            content = img_to_tensor(cv2.cvtColor(padding(self.content_img, 32), cv2.COLOR_BGR2RGB)).to(DEVICE)
            c_feats = self.image_encoder(content)
            s_feats = self.image_encoder(style)
            c_adain_feat_3 = self.ada_attn_3(c_feats[2], s_feats[2], get_key(c_feats, 2), get_key(s_feats, 2))
            cs = self.transformer(c_feats[3], s_feats[3], c_feats[4], s_feats[4], get_key(c_feats, 3), get_key(s_feats, 3),
                             get_key(c_feats, 4), get_key(s_feats, 4))
            cs = self.decoder(cs, c_adain_feat_3)
            cs = tensor_to_img(cs[:, :, :int(self.content_shape[0]), :int(self.content_shape[1])])
            cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)

        if resize:
            cs = cv2.resize(cs, (int(self.content_shape[1]), int(self.content_shape[0])))
            self.content_img = cv2.resize(self.content_img, (int(self.content_shape[1]), int(self.content_shape[0])))

        if mask_path:
            cs = cs * self.seg_mask + self.content_img * (1 - self.seg_mask)

        return cs


if __name__ == '__main__':
    args = infer_args()
    result = InferStyleTransfer(args.checkpoint_path).run(args.content_path, args.mask_path, args.style_path, args.resize, args.keep_ratio)
    cv2.imwrite("output/result.png", result)
    # cv2.waitKey(0)