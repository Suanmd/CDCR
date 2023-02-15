import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
from models import register
from utils import make_coord
import math

@register('cdcr2')
class CDCR2(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None, imnet_spec2=None,
                 local_ensemble=True, feat_unfold=True, align_corners=True,
                 dense_predict_with_for=True, num_experts=10, ksize=3,
                 weights_fused_with_mul=False, self_do_step_1=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.align_corners = align_corners # False in Liif
        self.for_flag = dense_predict_with_for # need to choose outdim: 3(with true) or 27(with false)
        self.mul_flag = weights_fused_with_mul
        self.step1 = self_do_step_1

        self.encoder = models.make(encoder_spec)
        self.num_experts = num_experts
        self.ksize = ksize # choose 1 or 3

        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(3, 3, ksize, ksize)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        # FC layers to generate routing weights
        self.routing = nn.Sequential(nn.Linear(2, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, num_experts),
                                     nn.Softmax(1))

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            imnet_in_dim += 2 # cell decode
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

        if imnet_spec2 is not None:
            imnet2_in_dim = 8
            imnet2_in_dim += 2 # cell decode
            self.imnet2 = models.make(imnet_spec2, args={'in_dim': imnet2_in_dim})
        else:
            self.imnet2 = None

    def softmax(self, x):
        row_max = np.max(x)
        x = x - row_max
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        s = x_exp / x_sum
        return s

    def guass_with_central(self, central=1.0, sig=0.5, data_min=1.0, data_max=999.0, sample_numbers=10):
        # scale_factor from 1 to 4 (default)
        # o  o  o  o  o  o  o  o  o  o
        # x1       x2       x3       x4
        data = np.linspace(data_min, data_max, sample_numbers)
        sqrt_2pi = np.power(2 * np.pi, 0.5)
        coef = 1 / (sqrt_2pi * sig)
        powercoef = -1 / (2 * np.power(sig, 2))
        mypow = powercoef * (np.power((data - central), 2))
        return self.softmax(coef * (np.exp(mypow)))

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=self.align_corners)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rets = []
        conv_pred_inp = []
        if self.for_flag:
            q_lists = [-1, 0, 1]
        else:
            q_lists = [0]
        for q1 in q_lists:
            for q2 in q_lists:
                preds = []
                areas = []
                for vx in vx_lst:
                    for vy in vy_lst:
                        coord_ = coord.clone()
                        coord_[:, :, 0] += vx * rx + q1 * torch.mean(cell, dim=1, keepdim=False)[:, 0] \
                            .reshape(coord.shape[0], 1).contiguous().repeat(1, coord.shape[1]) + eps_shift
                        coord_[:, :, 1] += vy * ry + q2 * torch.mean(cell, dim=1, keepdim=False)[:, 1] \
                            .reshape(coord.shape[0], 1).contiguous().repeat(1, coord.shape[1]) + eps_shift
                        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                        q_feat = F.grid_sample(
                            feat, coord_.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=self.align_corners)[:, :, 0, :] \
                            .permute(0, 2, 1)
                        q_coord = F.grid_sample(
                            feat_coord, coord_.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=self.align_corners)[:, :, 0, :] \
                            .permute(0, 2, 1)
                        rel_coord = coord - q_coord
                        rel_coord[:, :, 0] *= feat.shape[-2]
                        rel_coord[:, :, 1] *= feat.shape[-1]
                        inp = torch.cat([q_feat, rel_coord], dim=-1)


                        rel_cell = cell.clone()
                        rel_cell[:, :, 0] *= feat.shape[-2]
                        rel_cell[:, :, 1] *= feat.shape[-1]
                        inp = torch.cat([inp, rel_cell], dim=-1)

                        bs, q = coord.shape[:2]
                        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                        preds.append(pred)

                        if q1 == 0 and q2 == 0: # central
                            conv_pred_inp.append(rel_coord)

                        area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                        areas.append(area + 1e-9)

                tot_area = torch.stack(areas).sum(dim=0)
                if self.local_ensemble:
                    t = areas[0]; areas[0] = areas[3]; areas[3] = t
                    t = areas[1]; areas[1] = areas[2]; areas[2] = t
                ret = 0
                for pred, area in zip(preds, areas):
                    ret = ret + pred * (area / tot_area).unsqueeze(-1)
                rets.append(ret)

        retsp = torch.stack(rets, dim=-1).squeeze(-1)
        bs, q = retsp.shape[:2]
        retsp = retsp.view(bs, q, 3, 9)  # [batchsize, sample_q, 3, 9]

        # step 1
        # scaling_factors -> torch.round(2 / cell[:, 0, 0]) / 48
        if self.step1:
            _rs = []
            for b in range(bs):
                routing_weights_0 = self.routing(rel_cell[b, 0, :].unsqueeze(0)).view(self.num_experts, 1, 1)
                routing_weights_1 = self.guass_with_central(central=(torch.round(2 / cell[b, 0, 0]) / feat.shape[-1]).item(), sample_numbers=self.num_experts)
                routing_weights_1 = torch.from_numpy(routing_weights_1.astype(np.float32)).unsqueeze(-1).unsqueeze(-1).cuda()
                if self.mul_flag:
                    routing_weights = routing_weights_0 * routing_weights_1 * 10
                else:
                    routing_weights = routing_weights_0 + routing_weights_1
                # fuse experts
                fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
                fused_weight = fused_weight.view(-1, 3, self.ksize, self.ksize)
                if self.ksize == 3:
                    _rs.append(F.conv2d(retsp[b].view(q, 3, 3, 3), fused_weight, None, stride=1, padding=1))
                elif self.ksize == 1:
                    _rs.append(F.conv2d(retsp[b].view(q, 3, 3, 3), fused_weight, None, stride=1, padding=0))
                else:
                    raise
            retsp_residual = torch.stack(_rs).reshape(bs, q, 3, -1)
            retsp = retsp + retsp_residual

        # step 2
        conv_pred_inp = torch.cat(conv_pred_inp, dim=-1)
        conv_pred_inp = torch.cat((conv_pred_inp, rel_cell), dim=-1)
        conv_pred = self.imnet2(conv_pred_inp.view(bs * q, -1)).view(bs, q, -1)  # [batchsize, sample_q, 9]
        theoutput = torch.bmm(retsp.view(bs*q, retsp.shape[2], -1),
                               conv_pred.view(bs*q, conv_pred.shape[2], -1)).view(bs, q, -1)  # [batchsize, sample_q, 3]
        return theoutput

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
