import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
import math
import numpy as np

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)

    return output

@register('arbsr')
class ArbSR(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,
                 channels=64, num_experts=4, bias=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(self.channels//8, self.channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(self.channels, self.channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

        # tail module
        modules_tail = [
            None,
            nn.Conv2d(64, 3, 3, 1, 1, bias=True)]
        self.tail = nn.Sequential(*modules_tail)


    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        # input : coord  [8, 2304, 2]
        #         cell   [8, 2304, 2]
        # outputs : pred [8, 2304, 3]

        feat = self.feat # [8, 64, 48, 48]
        b, c, h, w = feat.size()
        # scaling_factors -> torch.round(2 / cell[:, 0, 0]) / 48
        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)

        outputs = []

        for _b in range(b):
            # (1) coordinates in LR space
            ## coordinates in HR space
            scale = (torch.round(2 / cell[_b, 0, 0]) / h).item()
            coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(feat.device),
                       torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(feat.device)]

            ## coordinates in LR space
            coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
            coor_h = coor_h.permute(1, 0)
            coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

            input = torch.cat((
                torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
                torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
                coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
                coor_w.expand([round(scale * h), -1]).unsqueeze(0)
            ), 0).unsqueeze(0)

            # (2) predict filters and offsets
            embedding = self.body(input)
            ## offsets
            offset = self.offset(embedding)

            ## filters
            routing_weights = self.routing(embedding)
            routing_weights = routing_weights.view(self.num_experts, round(scale * h) * round(scale * w)).transpose(0, 1)  # (h*w) * n

            weight_compress = self.weight_compress.view(self.num_experts, -1)
            weight_compress = torch.matmul(routing_weights, weight_compress)
            weight_compress = weight_compress.view(1, round(scale * h), round(scale * w), self.channels // 8, self.channels)

            weight_expand = self.weight_expand.view(self.num_experts, -1)
            weight_expand = torch.matmul(routing_weights, weight_expand)
            weight_expand = weight_expand.view(1, round(scale * h), round(scale * w), self.channels, self.channels // 8)

            # (3) grid sample & spatially varying filtering
            ## grid sample
            fea0 = grid_sample(feat[_b].unsqueeze(0), offset, scale, scale)  ## 1 * h * w * c * 1
            fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)  ## 1 * h * w * c * 1

            ## spatially varying filtering
            out = torch.matmul(weight_compress.expand([1, -1, -1, -1, -1]), fea)
            out = torch.matmul(weight_expand.expand([1, -1, -1, -1, -1]), out).squeeze(-1)
            outs = out.permute(0, 3, 1, 2).contiguous() + fea0   # [1, 64, h, w]
            pred = self.tail[1](outs)  # [1, 3, h, w]

            ## sample the results
            q_res = F.grid_sample(pred, coord_q[_b].unsqueeze(0).flip(-1).unsqueeze(1), mode='nearest', align_corners=True)[:, :, 0, :].permute(0, 2, 1)
            outputs.append(q_res.squeeze())

        return torch.stack(outputs)

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
