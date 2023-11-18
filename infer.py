import cv2
import torch
import numpy as np

from utils import *
from models.decoder import Decoder
from models.vgg_encoder import Encoder
from models.AdaAttN import AdaAttN, Transformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StyleTransfer:

    def __init__(self):

        self.content_img = None
        self.content_mask = None
        self.content_shape = None

        self.style_img = None
        self.style_mask = None

        self.seg_mask = None

        self.image_encoder = Encoder().to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.ada_attn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=64 * 64).to(DEVICE)
        self.transformer = Transformer(in_planes=512, key_planes=512 + 256 + 128 + 64).to(DEVICE)

    def build_models(self, checkpoint_path):

        # self.image_encoder.load_state_dict(torch.load(checkpoint_path+'/vgg_normalised.pth'))
        self.transformer.load_state_dict(torch.load(checkpoint_path+'/latest_net_transformer.pth'))
        self.decoder.load_state_dict(torch.load(checkpoint_path+'/latest_net_decoder.pth'))
        self.ada_attn_3.load_state_dict(torch.load(checkpoint_path+'/latest_net_adaattn_3.pth'))

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

        ######################### Manually creating masks #########################
        _mask = np.ones_like(self.content_img)*255
        self.content_mask = np.expand_dims(_mask.transpose()[0], axis=2)
        _mask = np.ones_like(self.style_img)*255
        self.style_mask = np.expand_dims(_mask.transpose()[0], axis=2)
        ###########################################################################

        if resize:
            self.content_img = resize_img(self.content_img, 512, keep_ratio)
            self.style_img = resize_img(self.style_img, 512, keep_ratio)

    def run(self, content_path, mask_path, style_path, checkpoint_path, resize=True, keep_ratio=True):

        self.load_images(content_path, mask_path, style_path, resize, keep_ratio)
        self.build_models(checkpoint_path)

        with torch.no_grad():
            style = img_to_tensor(cv2.cvtColor(padding(self.style_img, 32), cv2.COLOR_BGR2RGB)).to(DEVICE)
            content = img_to_tensor(cv2.cvtColor(padding(self.content_img, 32), cv2.COLOR_BGR2RGB)).to(DEVICE)
            c_masks = [torch.from_numpy(padding(self.content_mask, 32)).unsqueeze(0).permute(0, 3, 1, 2).float().to(DEVICE)]
            s_masks = [torch.from_numpy(padding(self.style_mask, 32)).unsqueeze(0).permute(0, 3, 1, 2).float().to(DEVICE)]
            c_feats = self.image_encoder(content)
            s_feats = self.image_encoder(style)
            c_adain_feat_3 = self.ada_attn_3(c_feats[2], s_feats[2], get_key(c_feats, 2), get_key(s_feats, 2), None,
                                        c_masks, s_masks)
            cs = self.transformer(c_feats[3], s_feats[3], c_feats[4], s_feats[4], get_key(c_feats, 3), get_key(s_feats, 3),
                             get_key(c_feats, 4), get_key(s_feats, 4), None, c_masks, s_masks)
            cs = self.decoder(cs, c_adain_feat_3)
            cs = tensor_to_img(cs[:, :, :self.content_shape[0], :self.content_shape[1]])
            cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)

        if resize:
            cs = cv2.resize(cs, (self.content_shape[1], self.content_shape[0]))
            self.content_img = cv2.resize(self.content_img, (self.content_shape[1], self.content_shape[0]))

        if mask_path:
            cs = cs * self.seg_mask + self.content_img * (1 - self.seg_mask)

        return cs


if __name__ == '__main__':
    args = setup_args()
    result = StyleTransfer().run(args.content_path, args.mask_path, args.style_path, args.checkpoint_path, args.resize, args.keep_ratio)
    cv2.imshow("result", result)
    cv2.waitKey(0)