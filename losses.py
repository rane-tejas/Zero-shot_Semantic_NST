import torch
import torch.nn as nn

from utils import *

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LossFunctions:

    def __init__(self, lambda_content=1., lambda_global=1., lambda_local=1.):

        self.lambda_content = lambda_content
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.criterion = nn.MSELoss().to(DEVICE)
        self.max_sample = 64 * 64

    def content_loss(self, stylized_features, content_features):

        loss = torch.tensor(0., device=DEVICE)
        if self.lambda_content > 0:
            for i in range(1, 5):
                loss += self.criterion(mean_variance_norm(stylized_features[i]),
                                       mean_variance_norm(content_features[i]))
        return loss

    def style_loss(self, stylized_features, content_features, style_features):

        # Global Style Loss
        loss_global = torch.tensor(0., device=DEVICE)
        if self.lambda_global > 0:
            for i in range(1, 5):
                style_features_mean, style_features_std = calc_mean_std(style_features[i])
                stylized_features_mean, stylized_features_std = calc_mean_std(stylized_features[i])
                loss_global += self.criterion(stylized_features_mean, style_features_mean) \
                            + self.criterion(stylized_features_std, style_features_std)

        # Local Feature Loss
        loss_local = torch.tensor(0., device=DEVICE)
        if self.lambda_local > 0:
            for i in range(1, 5):

                #Defining Queries, Keys and Values
                queries = get_key(content_features, i, need_shallow=True)
                keys = get_key(style_features, i, need_shallow=True)
                values = style_features[i]

                b, _, h_k, w_k = keys.size()
                keys = keys.view(b, -1, h_k * w_k).contiguous()
                if h_k * w_k > self.max_sample:
                    index = torch.randperm(h_k * w_k).to(DEVICE)[:self.max_sample]
                    keys = keys[:, :, index]
                    values_flattened = values.view(b, -1, h_k * w_k)[:, :, index].transpose(1, 2).contiguous()
                else:
                    values_flattened = values.view(b, -1, h_k * w_k).transpose(1, 2).contiguous()
                b, _, h_q, w_q = queries.size()
                queries = queries.view(b, -1, h_q * w_q).permute(0, 2, 1).contiguous()

                # Attention = softmax(queries * keys)
                attn = torch.bmm(queries, keys)
                attn = torch.softmax(attn, dim=-1)

                # Attention-weighted Mean = values * Attention
                mean = torch.bmm(attn, values_flattened)

                # Attention-weighted Std = sqrt(Attention * values^2 - Mean^2)
                std = torch.sqrt(torch.relu(torch.bmm(attn, values_flattened ** 2) - mean ** 2))

                mean = mean.view(b, h_q, w_q, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_q, w_q, -1).permute(0, 3, 1, 2).contiguous()

                loss_local += self.criterion(stylized_features[i], std * mean_variance_norm(content_features[i]) + mean)

        return loss_global + loss_local