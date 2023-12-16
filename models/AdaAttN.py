import torch
import torch.nn as nn

from utils import mean_variance_norm

torch.manual_seed(42)

class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None, checkpoint_path=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.max_sample = max_sample

        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path+'/adaattn.pth'))

    def forward(self, content, style, content_key, style_key):

        #Defining Queries, Keys and Values
        F = self.f(content_key) # Queries
        G = self.g(style_key) # Keys
        H = self.h(style) # Values

        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)

        # Attention = softmax(queries * keys)
        A = torch.bmm(F, G)
        A = self.softmax(A)

        # Attention-weighted Mean = values * Attention
        mean = torch.bmm(A, style_flat)

        # Attention-weighted Std = sqrt(Attention * values^2 - Mean^2)
        std = torch.sqrt(torch.relu(torch.bmm(A, style_flat ** 2) - mean ** 2))

        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return std * mean_variance_norm(content) + mean


class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None, checkpoint_path=None):
        super(Transformer, self).__init__()

        self.ada_attn_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.ada_attn_5_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes + 512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path+'/transformer.pth'))

    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key,
                content5_1_key, style5_1_key):
        return self.merge_conv(self.merge_conv_pad(
            self.ada_attn_4_1(content4_1, style4_1, content4_1_key, style4_1_key) +
            self.upsample5_1(self.ada_attn_5_1(content5_1, style5_1, content5_1_key, style5_1_key))))