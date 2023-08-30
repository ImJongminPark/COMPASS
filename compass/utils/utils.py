import os

import torch
import torch.optim as optim

import numpy as np
from PIL import Image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_optimizers(net, cfg):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=cfg['lr'],
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=cfg['lr_aux'],
    )
    return optimizer, aux_optimizer


def output_img_save(output, epoch, count, scale_idx, image_save_path, name):
    output = output.squeeze(0).detach().cpu().clone().numpy()
    output *= 255.0
    output = output.clip(0, 255)
    ts = (1, 2, 0)
    output = output.transpose(ts)

    out = Image.fromarray(np.uint8(output), mode='RGB')
    out.save(image_save_path + '/output_{:03d}_{:02d}_{:02d}_{}.png'.format(epoch, count, scale_idx, name))


def save_checkpoint(state, is_best, lmbda, epoch):
    dir_name = "checkpoints/lambda_" + str(lmbda)
    os.makedirs(dir_name, exist_ok=True)
    
    file_name = "epoch_" + format(epoch, '04') + ".pth.tar"
    
    if epoch % 10 == 0:
        torch.save(state, dir_name + '/' + file_name)
    
    if is_best:
        torch.save(state, dir_name + '/' + "best_model.pth.tar")

