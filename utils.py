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

def get_key(feats, last_layer_idx):
    results = []
    _, _, h, w = feats[last_layer_idx].shape
    for i in range(last_layer_idx):
        results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
    results.append(mean_variance_norm(feats[last_layer_idx]))
    return torch.cat(results, dim=1)

def setup_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--content_path", type=str, default="data/content/c1.jpg",
                        help="Path to a single content img")
    parser.add_argument("-s", "--style_path", type=str, default="data/style/vg_starry_night.jpg",
                        help="Path to a single style img")
    parser.add_argument("--checkpoint_path", type=str, default="ckpt",
                        help="Path to the checkpoint drectory")
    parser.add_argument("-o", "--output_dir", type=str, default='output/',
                        help="Output path")
    parser.add_argument("--resize", action='store_true',
                        help="Whether resize images to the 512 scale, which is the training resolution "
                            "of the model and may yield better performance")
    parser.add_argument("--keep_ratio", action='store_true',
                        help="Whether keep the aspect ratio of original images while resizing")

    return parser.parse_args()