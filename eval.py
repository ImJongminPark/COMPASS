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
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time

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


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return img


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


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def continuous_SR(model, img, h, w, device):
    bs = img.shape[0]

    coord = make_coord((h, w)).to(device)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).to(device),
        coord.unsqueeze(0).repeat(bs, 1, 1), cell.unsqueeze(0).repeat(bs, 1, 1), bsize=30000)

    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(bs, h, w, 3).permute(0, 3, 1, 2)

    return pred


def output_img_save(output, image_save_path, img_num, name):
    output = output.squeeze(0).detach().cpu().clone().numpy()
    output *= 255.0
    output = output.clip(0, 255)
    ts = (1, 2, 0)
    output = output.transpose(ts)

    out = Image.fromarray(np.uint8(output), mode='RGB')
    out.save(image_save_path + '/output_{:02d}_{}.png'.format(img_num, name))


@torch.no_grad()
def inference(model, model_res, model_conti, x_multiscale, device, img_num, scale_1, scale_2):
    output_dict = {}

    bit_amount_list = []
    output_list = []
    conti_sr_time_list = []

    coord_list = []
    cell_list = []

    flops_total = 0
    encoding_time_total = 0

    flops_conti_sr = 0

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

    #num_pixels = x_multiscale[0].size(0) * x_multiscale[0].size(2) * x_multiscale[0].size(3)
    bit_total = sum(len(s[0]) for s in out_enc_base["strings"]) * 8.0
    bit_amount_list.append(bit_total)

    for i in range(1, len(x_multiscale)):
        start_time = time.time()
        conti_sr = model_conti(out_comp, coord_list[i-1], cell_list[i-1])
        conti_sr_time_list.append(time.time() - start_time)
        residual = x_multiscale[i] - conti_sr

        encoding_start_time = time.time()
        out_enc = model_res.compress(residual)
        encoding_time = time.time() - encoding_start_time
        encoding_time_total += encoding_time

        out_dec = model_res.decompress(out_enc["strings"], out_enc["shape"])
        out_comp = out_dec['x_hat'] + conti_sr
        output_list.append(out_comp)
        #num_pixels = x_multiscale[i].size(0) * x_multiscale[i].size(2) * x_multiscale[i].size(3)
        bit_total += sum(len(s[0]) for s in out_enc["strings"]) * 8.0
        bit_amount_list.append(bit_total)

        flops, params = profile(model_conti, inputs=(out_comp, coord_list[i-1], cell_list[i-1],))
        #flops_conti_sr += flops

        #for name, module in model_conti.named_modules():
        #    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        #        module_flops = flops / 1e9
        #        print(f"{name}: {module_flops:.3f} G FLOPs")

        #flops, params = profile(model_res, inputs=(residual,))
        #flops_total += flops

    time_overall = time.time() - start_time_overall

    image_save_path = './results_9_scale'
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
            v_conti_time = conti_sr_time_list[i-1]

            output_dict[k_conti_time] = v_conti_time

        output_img_save(x_multiscale[i], image_save_path, img_num, '_' + str(i) + '_GT')
        output_img_save(output_list[i], image_save_path, img_num, '_' + str(i) + '_PSNR_' + str(v_psnr) + '_bit_' + str(v_bit_amount))

    k_FLOPS = 'FLOPS (conti. sr)'
    v_FLOPS = flops_conti_sr

    output_dict[k_FLOPS] = v_FLOPS

    k_total_time = 'total_time'
    v_total_time = time_overall

    output_dict[k_total_time] = v_total_time

    k_total_time_ex_enc = 'total_time (enc x)'
    v_total_time_ex_enc = time_overall - encoding_time_total

    output_dict[k_total_time_ex_enc] = v_total_time_ex_enc

    return output_dict


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr": psnr(x, out_net["x_hat"]),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def load_checkpoint(arch1: str, arch2: str, checkpoint_path: str) -> nn.Module:
    state_dict_base = load_state_dict(torch.load(checkpoint_path))['base_state_dict']
    state_dict_res = load_state_dict(torch.load(checkpoint_path))['residual_state_dict']
    return architectures[arch1].from_state_dict(state_dict_base).eval(), architectures[arch2].from_state_dict(state_dict_res).eval()


def eval_model(args, model, model_res, filepaths, scale_layer_1, scale_layer_2, entropy_estimation=False, half=False):
    device = torch.device('cuda')
    metrics = defaultdict(float)

    model_conti = LIFF_prediction(args)
    checkpoint = torch.load(args.paths[0], map_location=device)
    model_conti.load_state_dict(checkpoint["conti_sr_state_dict"])

    count = 0

    for f in filepaths:
        count += 1
        x = read_image(f)

        scale_layer1 = 0.5
        scale_layer2 = 1.0

        scale_list = [0.25, scale_layer1, scale_layer2]
 
        size_list = [(int(x.width * scale), int(x.height * scale)) for scale in scale_list]

        x_multiscale = []
        for size in size_list:
            x_multiscale.append(transforms.ToTensor()(x.resize(size)).to(device))

        if not entropy_estimation:
            if half:
                model = model.half()
                model_res = model_res.half()
                x = x.half()
            model.to(device)
            model_res.to(device)
            model_conti.to(device)
            rv = inference(model, model_res, model_conti, x_multiscale, device, img_num=count, scale_1=scale_layer_1, scale_2=scale_layer_2)
        else:
            rv = inference_entropy_estimation(model, x)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(
        add_help=False,
    )
    # Common options.
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-ab",
        "--architecture_base",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture (base)",
        required=True,
    )
    parent_parser.add_argument(
        "-ar",
        "--architecture_res",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture (res)",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    parser.add_argument('--G0', type=int, default=64,
                        help='default number of filters. (Use in RDN)')
    parser.add_argument('--RDNkSize', type=int, default=3,
                        help='default kernel size. (Use in RDN)')
    parser.add_argument('--RDNconfig', type=str, default='D',
                        help='parameters config of RDN. (Use in RDN)')

    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )
    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    checkpoint_parser.add_argument(
        "-s1",
        type=float,
        default=0.5,
        help="scale for layer 1"
    )
    checkpoint_parser.add_argument(
        "-s2",
        type=float,
        default=1.0,
        help="scale for layer 2"
    )

    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    filepaths = sorted(collect_images(args.dataset))
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    if args.source == "pretrained":
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    elif args.source == "checkpoint":
        runs = args.paths
        opts = (args.architecture_base, args.architecture_res)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model, model_res = load_func(*opts, run)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
            model_res = model_res.to("cuda")
        metrics = eval_model(args, model, model_res, filepaths, args.s1, args.s2, args.entropy_estimation, args.half)
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name(base)": args.architecture_base,
        "name(res)": args.architecture_res,
        "description": f"Inference ({description})",
        "results": results,
    }

    header1 = ["name(base)", "name(res)"]
    data1 = [args.architecture_base, args.architecture_res]

    header2 = list(output["results"].keys())
    data2 = list(itertools.chain.from_iterable(list(output["results"].values())))

    with open('./bit_amount_PSNR_results_scale_1_' + str(args.s1) + '_scale_2_' + str(args.s2) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(data1)
        writer.writerow(header2)
        writer.writerow(data2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
