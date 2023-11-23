import torch
import torch.nn as nn

from utils import *

torch.manual_seed(42)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

class LossFunctions:

    def __init__(self, lambda_content=1., lambda_global=1., lambda_local=1.):

        self.lambda_content = lambda_content
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.criterion = nn.MSELoss().to(DEVICE)
        self.max_sample = 64 * 64 # TODO

    def content_loss(self, stylized_feats, content_feats):
        loss = torch.tensor(0., device=DEVICE)
        if self.lambda_content > 0:
            for i in range(1, 5):
                loss += self.criterion(mean_variance_norm(stylized_feats[i]),
                                                       mean_variance_norm(content_feats[i]))
        return loss

    def style_loss(self, stylized_feats, content_feats, style_feats):
        loss_global = torch.tensor(0., device=DEVICE)
        if self.lambda_global > 0:
            for i in range(1, 5):
                s_feats_mean, s_feats_std = calc_mean_std(style_feats[i])
                stylized_feats_mean, stylized_feats_std = calc_mean_std(stylized_feats[i])
                loss_global += self.criterion(
                    stylized_feats_mean, s_feats_mean) + self.criterion(stylized_feats_std, s_feats_std)
        loss_local = torch.tensor(0., device=DEVICE)
        if self.lambda_local > 0:
            for i in range(1, 5):
                c_key = get_key(content_feats, i, need_shallow=True)
                s_key = get_key(style_feats, i, need_shallow=True)
                s_value = style_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                if h_s * w_s > self.max_sample:
                    index = torch.randperm(h_s * w_s).to(DEVICE)[:self.max_sample]
                    s_key = s_key[:, :, index]
                    style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
                else:
                    style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                loss_local += self.criterion(stylized_feats[i], std * mean_variance_norm(content_feats[i]) + mean)

        return loss_global + loss_local