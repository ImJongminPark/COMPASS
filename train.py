# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random
import time
import yaml

import sys
sys.path.append('./')

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder_train, ImageFolder_test
from compressai.zoo import models

from compass.model import *
from compass.utils.loss import *
from compass.utils.metric import *
from compass.utils.utils import *
from compass.utils.distribution import *

from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(wrap_model, criterion, train_dataloader, optimizer_el, aux_optimizer_el, optimizer_prediction, epoch, clip_max_norm, writer):
    wrap_model.module.model.train()
    wrap_model.module.model_el.train()
    wrap_model.module.model_prediction.train()
    device = next(wrap_model.module.model.parameters()).device

    train_loss = AverageMeter()
    mse_loss_1 = AverageMeter()
    mse_loss_2 = AverageMeter()
    bpp_loss_1 = AverageMeter()
    bpp_loss_2 = AverageMeter()

    loop_time = time.time()
    for i, d in enumerate(train_dataloader):
        img_base = d[0].to(device)  # [0, 1] normalized
        img_el1 = d[1].to(device)
        img_el2 = d[2].to(device)

        optimizer_el.zero_grad()
        aux_optimizer_el.zero_grad()
        optimizer_prediction.zero_grad()

        out1, out2, out_el1, out_el2, _, _ = wrap_model(img_base, img_el1, img_el2)

        out_criterion = criterion(out_el1, out_el2, out1, out2, img_el1, img_el2)

        train_loss.update(out_criterion["loss"].item())
        mse_loss_1.update(out_criterion["mse_loss_1"].item())
        mse_loss_2.update(out_criterion["mse_loss_2"].item())
        bpp_loss_1.update(out_criterion["bpp_loss_1"].item())
        bpp_loss_2.update(out_criterion["bpp_loss_2"].item())

        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(wrap_model.module.model.parameters(), clip_max_norm)
            torch.nn.utils.clip_grad_norm_(wrap_model.module.model_el.parameters(), clip_max_norm)
            torch.nn.utils.clip_grad_norm_(wrap_model.module.model_prediction.parameters(), clip_max_norm)

        optimizer_el.step()
        optimizer_prediction.step()

        aux_loss_el = wrap_model.module.model_el.aux_loss()
        aux_loss_el.backward()
        aux_optimizer_el.step()

        if i % 10 == 0:
            check_time = time.time() - loop_time
            if is_main_process():
                print(
                    f"Train epoch {epoch}: ["
                    f"{i}/{len(train_dataloader)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]\t"
                    f'Total Loss: {out_criterion["loss"].item():.5f}\t|'
                    f'MSE loss (EL1): {out_criterion["mse_loss_1"].item():.5f}\t|'
                    f'MSE loss (EL2): {out_criterion["mse_loss_2"].item():.5f}\t|'
                    f'Bpp loss (EL1): {out_criterion["bpp_loss_1"].item():.5f}\t|'
                    f'Bpp loss (EL2): {out_criterion["bpp_loss_2"].item():.5f}\t|'
                    f"Aux loss(EL): {aux_loss_el.item():.2f}\t|"
                    f"Time: {check_time:.2f}"
                )
            loop_time = time.time()

    writer.add_scalar(f"Train/train loss (Epoch)", train_loss.avg, (epoch + 1) * len(train_dataloader))
    writer.add_scalar(f"Train/MSE loss 1", mse_loss_1.avg, (epoch + 1) * len(train_dataloader))
    writer.add_scalar(f"Train/MSE loss 2", mse_loss_2.avg, (epoch + 1) * len(train_dataloader))
    writer.add_scalar(f"Train/BPP loss 1", bpp_loss_1.avg, (epoch + 1) * len(train_dataloader))
    writer.add_scalar(f"Train/BPP loss 2", bpp_loss_2.avg, (epoch + 1) * len(train_dataloader))

    return train_loss


def test_epoch(epoch, test_dataloader, wrap_model, criterion, save_name, train_loss, len_train_dataloader, writer, learning_rate_comp, learning_rate_conti):
    wrap_model.module.model.eval()
    wrap_model.module.model_el.eval()
    wrap_model.module.model_prediction.eval()

    device = next(wrap_model.module.model.parameters()).device

    loss_scale = []
    loss_scale_mse_1 = []
    loss_scale_mse_2 = []
    loss_scale_bpp_1 = []
    loss_scale_bpp_2 = []

    for i in range(0, 16):
        loss_scale.append(AverageMeter())
        loss_scale_mse_1.append(AverageMeter())
        loss_scale_mse_2.append(AverageMeter())
        loss_scale_bpp_1.append(AverageMeter())
        loss_scale_bpp_2.append(AverageMeter())

    mse_loss_1 = AverageMeter()
    mse_loss_2 = AverageMeter()
    bpp_loss_1 = AverageMeter()
    bpp_loss_2 = AverageMeter()

    loss = AverageMeter()
    aux_loss_el = AverageMeter()

    count = 0

    #image_save_path = './saved_images/' + save_name
    #os.makedirs(image_save_path, exist_ok=True)

    with torch.no_grad():
        for d_list in test_dataloader:
            count += 1
            scale_idx = 0

            for d in d_list:
                img_base = d[0].to(device)  # [0, 1] normalized
                img_el1 = d[1].to(device)
                img_el2 = d[2].to(device)

                out1, out2, out_el1, out_el2, el1, el2 = wrap_model(img_base, img_el1, img_el2)

                out_criterion = criterion(out_el1, out_el2, out1, out2, img_el1, img_el2)

                mse_loss_1.update(out_criterion["mse_loss_1"])
                mse_loss_2.update(out_criterion["mse_loss_2"])
                bpp_loss_1.update(out_criterion["bpp_loss_1"])
                bpp_loss_2.update(out_criterion["bpp_loss_2"])
                loss.update(out_criterion["loss"])

                loss_scale[scale_idx].update(out_criterion["loss"])
                loss_scale_mse_1[scale_idx].update(out_criterion["mse_loss_1"])
                loss_scale_mse_2[scale_idx].update(out_criterion["mse_loss_2"])
                loss_scale_bpp_1[scale_idx].update(out_criterion["bpp_loss_1"])
                loss_scale_bpp_2[scale_idx].update(out_criterion["bpp_loss_2"])

                aux_loss_el.update(wrap_model.module.model_el.aux_loss())

                scale_idx += 1

        #if epoch % 10 == 0:
        #    output_img_save(img_el1, epoch, count, scale_idx, image_save_path, '_div2_GT')
        #    output_img_save(img_el2, epoch, count, scale_idx, image_save_path, '_div1_GT')

        #    output_img_save(out1, epoch, count, scale_idx, image_save_path, '_div2')
        #    output_img_save(out2, epoch, count, scale_idx, image_save_path, '_div1')

        #    output_img_save(el1, epoch, count, scale_idx, image_save_path, '_el2_in')
        #    output_img_save(out_el1['x_hat'], epoch, count, scale_idx, image_save_path, '_el2_out')
        #    output_img_save(el2, epoch, count, scale_idx, image_save_path, '_el1_in')
        #    output_img_save(out_el2['x_hat'], epoch, count, scale_idx, image_save_path, '_el1_out')

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.5f} |"
        f'\tMSE loss 1: {mse_loss_1.avg:.5f} |'
        f'\tMSE loss 2: {mse_loss_2.avg:.5f} |'
        f'\tBpp loss 1: {bpp_loss_1.avg:.5f} |'
        f'\tBpp loss 2: {bpp_loss_2.avg:.5f} |'
        f"\tAux loss(Res): {aux_loss_el.avg:.2f}\n"
    )

    #testinfo = "Epoch: {} | Total(Test) loss:{} | Train loss:{} \n MSE loss 1:{}, MSE loss 2:{}\n"\
    #           "Bpp loss 1:{}, Bpp loss 2:{}\n, Aux loss(Res):{}, learning rate (comp):{}, learning rate (conti):{}\n"\
    #    .format(epoch, loss.avg, train_loss.avg, mse_loss_1.avg, mse_loss_2.avg,
    #            bpp_loss_1.avg, bpp_loss_2.avg, aux_loss_el.avg, learning_rate_comp, learning_rate_conti)

    #now = time.localtime()
    #testtime = "%04d/%02d/%02d %02d:%02d:%02d\n" % (
    #    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    #with open('log_file_' + save_name + '.txt', 'a+') as f:
    #    f.write(testinfo)
    #    f.write(testtime)

    writer.add_scalar(f"Val_total/test loss", loss.avg, (epoch + 1) * len_train_dataloader)
    writer.add_scalar(f"Val_MSE all/MSE loss 1", mse_loss_1.avg, (epoch + 1) * len_train_dataloader)
    writer.add_scalar(f"Val_MSE all/MSE loss 2", mse_loss_2.avg, (epoch + 1) * len_train_dataloader)
    writer.add_scalar(f"Val_BPP all/BPP loss 1", bpp_loss_1.avg, (epoch + 1) * len_train_dataloader)
    writer.add_scalar(f"Val_BPP all/BPP loss 2", bpp_loss_2.avg, (epoch + 1) * len_train_dataloader)

    writer.add_scalar(f"LR comp", learning_rate_comp, (epoch + 1) * len_train_dataloader)
    writer.add_scalar(f"LR conti. SR", learning_rate_conti, (epoch + 1) * len_train_dataloader)

    for i in range(len(loss_scale)):
        writer.add_scalar(f"Val_total/test loss for scale {i}", loss_scale[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val_MSE 1/MSE loss 1 for scale {i}", loss_scale_mse_1[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val_MSE 2/MSE loss 2 for scale {i}", loss_scale_mse_2[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val_BPP 1/BPP loss 1 for scale {i}", loss_scale_bpp_1[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val_BPP 2/BPP loss 2 for scale {i}", loss_scale_bpp_2[i].avg, (epoch + 1) * len_train_dataloader)

    return loss.avg


def test_epoch_RD(epoch, test_dataloader, wrap_model, len_train_dataloader, writer):
    wrap_model.module.model.update()
    wrap_model.module.model_el.update()

    wrap_model.module.model.eval()
    wrap_model.module.model_el.eval()
    wrap_model.module.model_prediction.eval()

    device = next(wrap_model.module.model.parameters()).device

    psnr_0 = []
    psnr_1 = []
    psnr_2 = []
    bit_0 = []
    bit_1 = []
    bit_2 = []

    count = 0
    for i in range(0, 16):
        psnr_0.append(AverageMeter())
        psnr_1.append(AverageMeter())
        psnr_2.append(AverageMeter())
        bit_0.append(AverageMeter())
        bit_1.append(AverageMeter())
        bit_2.append(AverageMeter())

    with torch.no_grad():
        for d_list in test_dataloader:
            count += 1
            scale_idx = 0
            for d in d_list:
                img_base = d[0].to(device)
                img_el1 = d[1].to(device)
                img_el2 = d[2].to(device)

                psnr_base, psnr_el1, psnr_el2, bit_base, bit_el1, bit_el2 = wrap_model.module.inference(img_base, img_el1, img_el2)

                psnr_0[scale_idx].update(psnr_base)
                psnr_1[scale_idx].update(psnr_el1)
                psnr_2[scale_idx].update(psnr_el2)

                bit_0[scale_idx].update(bit_base)
                bit_1[scale_idx].update(bit_el1)
                bit_2[scale_idx].update(bit_el2)

                scale_idx += 1

    for i in range(len(psnr_0)):
        writer.add_scalar(f"Val PSNR 0/PSNR 0 for scale {i}", psnr_0[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val PSNR 1/PSNR 1 for scale {i}", psnr_1[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val PSNR 2/PSNR 2 for scale {i}", psnr_2[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val Bit 0/Bit 0 for scale {i}", bit_0[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val Bit 1/Bit 1 for scale {i}", bit_1[i].avg, (epoch + 1) * len_train_dataloader)
        writer.add_scalar(f"Val Bit 2/Bit 2 for scale {i}", bit_2[i].avg, (epoch + 1) * len_train_dataloader)

    return


def main(cfg, args):
    save_name = "results"
    log_dir = "./log"
    
    writer = SummaryWriter(log_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
        random.seed(cfg['seed'])

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    synchronize()

    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder_train(cfg['dataset'], split=cfg['train_split'], transform=train_transforms)
    test_dataset = ImageFolder_test(cfg['dataset'], split=cfg['test_split'], transform=test_transforms)

    device = "cuda" if cfg['cuda'] and torch.cuda.is_available() else "cpu"

    multi_gpu_sampler_train = torch.utils.data.DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['batchSize'],
        num_workers=cfg['nWorkers'],
        sampler=multi_gpu_sampler_train,
        shuffle=(multi_gpu_sampler_train is None),
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg['nWorkers'],
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models[cfg['CompModel']['BL']](quality=cfg['quality'], pretrained=True, progress=True)
    net = net.to(device)
    net.update()

    net_enhance = models[cfg['CompModel']['EL']](quality=cfg['quality'], pretrained=True, progress=True)
    net_enhance = net_enhance.to(device)
    net_enhance.update()

    net_prediction = LIFF_prediction(cfg)
    net_prediction.to(device)

    if is_main_process():
        print("Lambda  : ", cfg['lmbda'])
        print("The number of parameters (Base Layer) : ", count_parameters(net))
        print("The number of parameters (Enhance. Layer) : ", count_parameters(net_enhance))
        print("The number of parameters (Inter-Layer Arbirtrary Scale Prediction) : ", count_parameters(net_prediction))

    last_epoch = 0

    # load coarse model

    checkpoint_el = os.path.join(cfg['checkpoint_el'], 'lambda_' + str(cfg['lmbda']), 'pretrained.pth.tar')
    pretrained_el = torch.load(checkpoint_el, map_location=device)
    pretrained_prediction = torch.load(cfg['checkpoint_prediction'], map_location=device)

    net_enhance.load_state_dict(pretrained_el["residual_state_dict"])
    net_enhance.update()

    net_prediction.load_state_dict(pretrained_prediction["model_state_dict"])

    if is_main_process():
        print("Loaded coarse model.")

    if cfg['checkpoint']:  # load from previous checkpoint
        print("Loading", cfg['checkpoint'])
        checkpoint = torch.load(cfg['checkpoint'], map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net_enhance.load_state_dict(checkpoint["residual_state_dict"])
        net_prediction.load_state_dict(checkpoint["conti_sr_state_dict"])

    wrap_model = torch.nn.parallel.DistributedDataParallel(COMPASS(net, net_enhance, net_prediction, cfg).to(device),
                                                           device_ids=[args.local_rank],
                                                           find_unused_parameters=True)

    optimizer_el, aux_optimizer_el, optimizer_prediction = wrap_model.module.optimizer()

    if cfg['checkpoint']:
        if is_main_process():
            print("Loading", cfg['checkpoint'])
        checkpoint = torch.load(cfg['checkpoint'], map_location=device)
        last_epoch = checkpoint["epoch"] + 1

        optimizer_el.load_state_dict(checkpoint["optimizer_res"])
        aux_optimizer_el.load_state_dict(checkpoint["aux_optimizer_res"])
        optimizer_prediction.load_state_dict(checkpoint["optimizer_conti_sr"])

    lr_scheduler_el = optim.lr_scheduler.StepLR(optimizer_el, step_size=cfg['optim']['step_size'], gamma=cfg['optim']['gamma'])
    lr_scheduler_prediction = optim.lr_scheduler.StepLR(optimizer_prediction, step_size=cfg['optim']['step_size'], gamma=cfg['optim']['gamma'])

    criterion = RateDistortionLoss(lmbda=cfg['lmbda'])

    best_loss = float("inf")
    for epoch in range(last_epoch, cfg['epochs']):
        if is_main_process():
            print(f"Learning rate (comp.) : {optimizer_el.param_groups[0]['lr']}")
            print(f"Learning rate (pred.) : {optimizer_prediction.param_groups[0]['lr']}")

        train_loss = train_one_epoch(
            wrap_model,
            criterion,
            train_dataloader,
            optimizer_el,
            aux_optimizer_el,
            optimizer_prediction,
            epoch,
            cfg['clip_max_norm'],
            writer
        )

        if is_main_process():
            loss = test_epoch(epoch, test_dataloader, wrap_model, criterion, save_name, train_loss,
                              len(train_dataloader), writer, learning_rate_comp=optimizer_el.param_groups[0]['lr'], learning_rate_conti=optimizer_prediction.param_groups[0]['lr'])
            if epoch % 10 == 0:
                test_epoch_RD(epoch, test_dataloader, wrap_model, len(train_dataloader), writer)

            lr_scheduler_el.step()
            lr_scheduler_prediction.step()

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if cfg['save']:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "base_state_dict": wrap_model.module.model.state_dict(),
                        "residual_state_dict": wrap_model.module.model_el.state_dict(),
                        "conti_sr_state_dict": wrap_model.module.model_prediction.state_dict(),
                        "loss": loss,
                        "optimizer_res": optimizer_el.state_dict(),
                        "aux_optimizer_res": aux_optimizer_el.state_dict(),
                        "optimizer_conti_sr": optimizer_prediction.state_dict(),
                        "lr_scheduler_el": lr_scheduler_el.state_dict(),
                    },
                    is_best,
                    cfg['lmbda'],
                    epoch
                )

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()

    with open('configs/cfg_train.yaml') as f:
        cfg = yaml.safe_load(f)

    main(cfg, args)
