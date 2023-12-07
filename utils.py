import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

def img_to_tensor(img):
    return (torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.).unsqueeze(0)

def tensor_to_img(img):
    return (img[0].data.cpu().numpy().transpose((1, 2, 0)).clip(0, 1) * 255 + 0.5).astype(np.uint8)

def resize_img(img, long_side=512, keep_ratio=True):
    if keep_ratio:
        h, w = img.shape[:2]
        if h < w:
            new_h = int(long_side * h / w)
            new_w = int(long_side)
        else:
            new_w = int(long_side * w / h)
            new_h = int(long_side)
        return cv2.resize(img, (new_w, new_h))
    else:
        return cv2.resize(img, (long_side, long_side))

def padding(img, factor=32):
    h, w = img.shape[:2]
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    new_img = np.zeros((h + pad_h, w + pad_w, img.shape[2]), dtype=img.dtype)
    new_img[:h, :w, :] = img
    return new_img

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def get_key(feats, last_layer_idx, need_shallow=True):
    if need_shallow and last_layer_idx > 0:
        results = []
        _, _, h, w = feats[last_layer_idx].shape
        for i in range(last_layer_idx):
            results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
        results.append(mean_variance_norm(feats[last_layer_idx]))
        return torch.cat(results, dim=1)
    else:
        return mean_variance_norm(feats[last_layer_idx])

def infer_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--content_path", type=str, default="data/content/c1.jpg",
                        help="Path to a single content img")
    parser.add_argument("-m", "--mask_path", type=str, default=None,
                        help="Path to a single mask img")
    parser.add_argument("-s", "--style_path", type=str, default="data/style/vg_starry_night.jpg",
                        help="Path to a single style img")
    parser.add_argument("--checkpoint_path", type=str, default="ckpt/pretrained",
                        help="Path to the checkpoint drectory")
    parser.add_argument("-o", "--output_dir", type=str, default='output/',
                        help="Output path")
    parser.add_argument("--resize", action='store_true',
                        help="Whether resize images to the 512 scale, which is the training resolution "
                            "of the model and may yield better performance")
    parser.add_argument("--keep_ratio", action='store_true',
                        help="Whether keep the aspect ratio of original images while resizing")

    return parser.parse_args()

def train_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_path", type=str, default="dataset/PhraseCut_mod",
                        help="Path to the dataset")
    parser.add_argument("-c", "--checkpoint_path", type=str, default="ckpt/pretrained",
                        help="Path to the checkpoint drectory")
    parser.add_argument("-l", "--log_dir", type=str, default='logs/',
                        help="Path to the log directory")
    parser.add_argument("-p", "--log_name", type=str, default='trial_logs',
                        help="name of the log file")
    parser.add_argument("-b", "--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("-e", "--num_epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")


    return parser.parse_args()


class Logger:

    import os
    import time
    import matplotlib.pyplot as plt

    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not self.os.path.exists(log_dir):
            self.os.makedirs(log_dir)

        self.plot_dir = self.os.path.join(log_dir, 'plots')
        if not self.os.path.exists(self.plot_dir):
            self.os.mkdir(self.plot_dir)
        self.log_file_path = self.log_dir + '/logs.txt'
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write('Logs date and time: '+self.time.strftime("%d-%m-%Y %H:%M:%S")+'\n\n')
        self.log_file = open(self.log_file_path, 'a')


        self.train_data = []
        self.val_data = []

    def log(self, tag, **kwargs):

        if tag == 'args':
            self.log_file.write('Training Args:\n')
            for k, v in kwargs.items():
                self.log_file.write(str(k)+': '+str(v)+'\n')
            self.log_file.write('#########################################################\n\n')

        elif tag == 'train':
            self.train_data.append([kwargs['loss']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Train Loss: {kwargs["loss"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'val':
            self.val_data.append([kwargs['loss']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Val Loss: {kwargs["loss"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'plot':
            self.plot(self.train_data, name='Train Loss', path=self.plot_dir)
            self.plot(self.val_data, name='Val Loss', path=self.plot_dir)

    def plot(self, data, name, path):

        self.plt.plot(data)
        self.plt.xlabel('Epochs')
        self.plt.ylabel(name)
        self.plt.title(name+' vs. Epochs')
        self.plt.savefig(self.os.path.join(path, name+'.png'), dpi=1200 ,bbox_inches='tight')
        self.plt.close()