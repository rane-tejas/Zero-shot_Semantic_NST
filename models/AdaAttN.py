import torch
import torch.nn as nn

from utils import mean_variance_norm

class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None, content_masks=None, style_masks=None):
        if content_masks is None:
            content_masks = []
        if style_masks is None:
            style_masks = []
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        style_masks = [nn.functional.interpolate(mask, size=(h_g, w_g), mode='nearest').view(
            b, 1, h_g * w_g).contiguous() for mask in style_masks]
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_masks = [mask[:, :, index] for mask in style_masks]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        content_masks = [nn.functional.interpolate(mask, size=(h, w), mode='nearest').view(
            b, 1, w * h).permute(0, 2, 1).contiguous() for mask in content_masks]
        S = torch.bmm(F, G)
        for content_mask, style_mask in zip(content_masks, style_masks):
            style_mask = 1. - style_mask
            attn_mask = torch.bmm(content_mask, style_mask)
            S = S.masked_fill(attn_mask.bool(), -1e15)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean


class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None):
        super(Transformer, self).__init__()
        self.ada_attn_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.ada_attn_5_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes + 512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key,
                content5_1_key, style5_1_key, seed=None, content_masks=None, style_masks=None):
        return self.merge_conv(self.merge_conv_pad(
            self.ada_attn_4_1(
                content4_1, style4_1, content4_1_key, style4_1_key, seed, content_masks, style_masks) +
            self.upsample5_1(self.ada_attn_5_1(
                content5_1, style5_1, content5_1_key, style5_1_key, seed, content_masks, style_masks))))