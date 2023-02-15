import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from utils import make_coord
from functools import reduce

import pdb

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def UpScaleModule(input, model, eval_bsize, norms, upscaling_factor=None, upscaling_sizes=None, flag=1):
    inp = (input - norms[0]) / norms[1]
    if upscaling_factor:
        _feat = F.interpolate(inp, scale_factor=upscaling_factor)
    elif upscaling_sizes:
        _feat = F.interpolate(inp, size=upscaling_sizes)
    else:
        raise Exception("Bad upscaling_factor or upscaling_sizes!")
    _coord = make_coord(_feat.shape[-2:], flatten=True).cuda().unsqueeze(0)
    _cell = torch.ones_like(_coord)
    _cell[:, :, 0] *= 2 / _feat.shape[-2]
    _cell[:, :, 1] *= 2 / _feat.shape[-1]
    if eval_bsize is None:
        with torch.no_grad():
            pred = model(inp, _coord, _cell)
    else:
        pred = batched_predict(model, inp, _coord, _cell, eval_bsize)
    pred = pred * norms[3] + norms[2]
    pred.clamp_(0, 1)
    ih, iw = inp.shape[-2:]
    s = math.sqrt(_coord.shape[1] / (ih * iw))
    shape = [inp.shape[0], round(ih * s), round(iw * s), 3]
    return pred.view(*shape).permute(0, 3, 1, 2).contiguous()

def UpScaleModule_with_various_scales(input, model, scales, eval_bsize, norms):
    _i = input
    _r = reduce(lambda x,y:x*y,scales)
    _x = round(input.shape[-2] * _r)
    _y = round(input.shape[-1] * _r)
    for _j, _s in enumerate(scales):
        if _j < len(scales) - 1:
            _x0 = round(_i.shape[-2] * _s)
            _y0 = round(_i.shape[-1] * _s)
            _i = UpScaleModule(input=_i, model=model, eval_bsize=eval_bsize, norms=norms, upscaling_sizes=(_x0,_y0))
        else: # the last element
            if _i.shape[-2]<_x and input.shape[-1]<_y:
                _i = UpScaleModule(input=_i, model=model, eval_bsize=eval_bsize, norms=norms, upscaling_sizes=(_x,_y))
            else:
                raise Exception("Bad Scales!")
    return _i

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, model_name=None, 
              verbose=False, save_img=True, flag_2_stage=True, flag_3_stage=True):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    number = 0
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        uprx = 2 / batch['cell'][0, :, -2].max() / batch['inp'].shape[-2]
        upry = 2 / batch['cell'][0, :, -1].max() / batch['inp'].shape[-1]
        upr = (uprx + upry) / 2
        pred = None
        scales = []

        scales.append([upr.item(),])

        if flag_2_stage:
            ths1 = 2.0
            ths1_min = 1.5
            ths1_max = 8.0
            alpha1 = 0.5  # 0.1
            _theta0 = torch.sqrt(upr)
            if _theta0 > ths1:
                scales.append([_theta0.item(), _theta0.item()])  # new added
            while _theta0 > ths1:
                _theta0 = _theta0 - alpha1 * torch.rand(1).cuda()
                if _theta0 > ths1_min and (upr / _theta0) < ths1_max:
                    scales.append([_theta0.item(), (upr / _theta0).item()])

        if flag_3_stage:
            ths2 = 2.0
            ths2_min = 1.5
            ths2_max = 8.0
            alpha2 = 1.0   # 0.5
            _theta1 = torch.pow(upr, 1/3)
            _theta2 = torch.pow(upr, 1/3)
            if _theta1 > ths2:
                scales.append([_theta1.item(), _theta1.item(), _theta1.item()])  # new added
            while _theta1 > ths2:
                _theta1 = _theta1 - alpha2 * torch.rand(1).cuda()
                if _theta1 > ths2_min and (upr / (_theta1*_theta2)) < ths2_max:
                    scales.append([_theta1.item(), _theta2.item(), (upr / (_theta1*_theta2)).item()])

        preds = []
        for element in scales:
            preds.append(UpScaleModule_with_various_scales(input=batch['inp'], 
                                                           model=model, 
                                                           scales=element, 
                                                           eval_bsize=eval_bsize, 
                                                           norms=[inp_sub, inp_div, gt_sub, gt_div]))
        pred = torch.mean(torch.stack(preds), dim=0)

        ih, iw = batch['inp'].shape[-2:]
        shape = [batch['inp'].shape[0], round(ih * upr.item()), round(iw * upr.item()), 3]
        batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()
        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), batch['inp'].shape[0])

        if save_img:
            _img = torch.clamp(torch.round(pred.squeeze(0).permute(1,2,0)*255), 0, 255).cpu().detach().numpy()
            model_name.split('/')[-2]
            save_dir = os.path.join('./results-SCA', 
                                    model_name.split('/')[-2], 
                                    config['test_dataset']['dataset']['args']['root_path'].split('/')[-2], 
                                    'x' + str(config['test_dataset']['wrapper']['args']['scale_min']))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_imdir = os.path.join(save_dir, str(number) + '_' + str(res.item())[:6] + '.jpg') 
            Image.fromarray(np.uint8(_img)).convert('RGB').save(save_imdir,"jpeg") 

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
        number += 1

    if save_img: # if True, then do
        os.rename(save_dir, save_dir + '_' + str(val_res.item())[:6])

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        model_name=args.model,
        verbose=True,
        save_img=True,
        flag_2_stage=True,
        flag_3_stage=True)
    print('result: {:.4f}'.format(res))
