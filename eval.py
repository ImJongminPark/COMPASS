# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import json
import math
import os
import sys
import time
import yaml

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from compressai.zoo import load_state_dict
from compressai.zoo import models as pretrained_models
from compressai.zoo.image import model_architectures as architectures

from compass.model import *
from compass.utils.metric import psnr


import numpy as np
import csv
import itertools

from thop import profile

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return img


def output_img_save(output, image_save_path, img_num, name):
    output = output.squeeze(0).detach().cpu().clone().numpy()
    output *= 255.0
    output = output.clip(0, 255)
    ts = (1, 2, 0)
    output = output.transpose(ts)

    out = Image.fromarray(np.uint8(output), mode='RGB')
    out.save(image_save_path + '/output_{:02d}_{}.png'.format(img_num, name))


@torch.no_grad()
def inference(model, model_el, model_prediction, x_multiscale, img_num):
    output_dict = {}

    bit_amount_list = []
    output_list = []
    prediction_time_list = []

    coord_list = []
    cell_list = []

    #flops_total = 0
    #flops_prediction = 0

    encoding_time_total = 0

    for i in range(len(x_multiscale)):
        x_multiscale[i] = x_multiscale[i].unsqueeze(0)

    for i in range(len(x_multiscale) - 1):
        img_coord = make_coord(x_multiscale[i+1].shape[-2::], flatten=False).cuda()
        img_coord = img_coord.permute(2, 0, 1).unsqueeze(0)
        img_coord = img_coord.expand(x_multiscale[i+1].shape[0], 2, *x_multiscale[i+1].shape[-2:])
        coord_list.append(img_coord)

        img_cell = torch.ones_like(img_coord)
        img_cell[:, 0] *= 2 / img_cell.size(2)
        img_cell[:, 1] *= 2 / img_cell.size(3)
        cell_list.append(img_cell)

    start_time_overall = time.time()

    encoding_start_time = time.time()
    out_enc_base = model.compress(x_multiscale[0])
    encoding_time_base = time.time() - encoding_start_time
    encoding_time_total += encoding_time_base

    out_dec_base = model.decompress(out_enc_base["strings"], out_enc_base["shape"])
    out_comp = out_dec_base['x_hat']
    output_list.append(out_comp)

    #flops, params = profile(model, inputs=(x_multiscale[0],))
    #flops_total += flops

    bit_total = sum(len(s[0]) for s in out_enc_base["strings"]) * 8.0
    bit_amount_list.append(bit_total)

    for i in range(1, len(x_multiscale)):
        start_time = time.time()
        prediction = model_prediction(out_comp, coord_list[i-1], cell_list[i-1])
        prediction_time_list.append(time.time() - start_time)
        residual = x_multiscale[i] - prediction

        encoding_start_time = time.time()
        out_enc = model_el.compress(residual)
        encoding_time = time.time() - encoding_start_time
        encoding_time_total += encoding_time

        out_dec = model_el.decompress(out_enc["strings"], out_enc["shape"])
        out_comp = out_dec['x_hat'] + prediction
        output_list.append(out_comp)
        bit_total += sum(len(s[0]) for s in out_enc["strings"]) * 8.0
        bit_amount_list.append(bit_total)

        #flops, params = profile(model_prediction, inputs=(out_comp, coord_list[i-1], cell_list[i-1],))
        #flops_prediction += flops

        #for name, module in model_prediction.named_modules():
        #    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        #        module_flops = flops / 1e9
        #        print(f"{name}: {module_flops:.3f} G FLOPs")

        #flops, params = profile(model_el, inputs=(residual,))
        #flops_total += flops

    time_overall = time.time() - start_time_overall

    image_save_path = './results_eval'
    os.makedirs(image_save_path, exist_ok=True)

    for i in range(len(x_multiscale)):
        k_psnr = f'psnr (stage {i})'
        v_psnr = psnr(x_multiscale[i], output_list[i].clamp_(0, 1))

        output_dict[k_psnr] = v_psnr

        k_bit_amount = f'bit_amount (stage {i})'
        v_bit_amount = bit_amount_list[i]

        output_dict[k_bit_amount] = v_bit_amount

        if i > 0:
            k_conti_time = f'conti. sr time (stage {i})'
            v_conti_time = prediction_time_list[i-1]

            output_dict[k_conti_time] = v_conti_time

        output_img_save(x_multiscale[i], image_save_path, img_num, '_' + str(i) + '_GT')
        output_img_save(output_list[i], image_save_path, img_num, '_' + str(i) + '_PSNR_' + str(v_psnr) + '_bit_' + str(v_bit_amount))

    #k_FLOPS = 'FLOPS (conti. sr)'
    #v_FLOPS = flops_prediction

    #output_dict[k_FLOPS] = v_FLOPS

    k_total_time = 'total_time'
    v_total_time = time_overall

    output_dict[k_total_time] = v_total_time

    k_total_time_ex_enc = 'total_time (w/o enc)'
    v_total_time_ex_enc = time_overall - encoding_time_total

    output_dict[k_total_time_ex_enc] = v_total_time_ex_enc

    return output_dict


def load_checkpoint(arch1: str, arch2: str, checkpoint_path: str) -> nn.Module:
    state_dict_base = load_state_dict(torch.load(checkpoint_path))['base_state_dict']
    state_dict_res = load_state_dict(torch.load(checkpoint_path))['residual_state_dict']
    return architectures[arch1].from_state_dict(state_dict_base).eval(), architectures[arch2].from_state_dict(state_dict_res).eval()


def eval_model(cfg, model, model_el, filepaths, scale_el1, scale_el2):
    device = torch.device('cuda')
    metrics = defaultdict(float)

    model_prediction = LIFF_prediction(cfg)
    checkpoint = torch.load(cfg['checkpoint'], map_location=device)
    model_prediction.load_state_dict(checkpoint["conti_sr_state_dict"])

    count = 0

    for f in filepaths:
        count += 1
        x = read_image(f)

        scale_list = [0.25, scale_el1, scale_el2]
 
        size_list = [(int(x.width * scale), int(x.height * scale)) for scale in scale_list]

        x_multiscale = []
        for size in size_list:
            x_multiscale.append(transforms.ToTensor()(x.resize(size)).to(device))

        model.to(device)
        model_el.to(device)
        model_prediction.to(device)
        rv = inference(model, model_el, model_prediction, x_multiscale, img_num=count)

        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def main(cfg):
    filepaths = sorted(collect_images(cfg['dataset']))
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(compressai.available_entropy_coders()[0])

    runs = [cfg['checkpoint']]
    opts = (cfg['CompModel']['BL'], cfg['CompModel']['EL'])
    load_func = load_checkpoint

    results = defaultdict(list)
    for run in runs:
        model, model_el = load_func(*opts, run)
        if cfg['cuda'] and torch.cuda.is_available():
            model = model.to("cuda")
            model_el = model_el.to("cuda")
        metrics = eval_model(cfg, model, model_el, filepaths, cfg['scale_e1'], cfg['scale_e2'])
        
        for k, v in metrics.items():
            results[k].append(v)

    description = (
        compressai.available_entropy_coders()[0]
    )

    output = {
        "name(base)": cfg['CompModel']['BL'],
        "name(res)": cfg['CompModel']['EL'],
        "description": f"Inference ({description})",
        "results": results,
    }

    header1 = ["name(base)", "name(res)"]
    data1 = [cfg['CompModel']['BL'], cfg['CompModel']['EL']]

    header2 = list(output["results"].keys())
    data2 = list(itertools.chain.from_iterable(list(output["results"].values())))

    with open('./bit_amount_PSNR_results_scale_1_' + str(cfg['scale_e1']) + '_scale_2_' + str(cfg['scale_e2']) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(data1)
        writer.writerow(header2)
        writer.writerow(data2)

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    with open('configs/cfg_eval.yaml') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
