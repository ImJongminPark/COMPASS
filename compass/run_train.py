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

import sys
sys.path.append('./')

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder_down_img_TRAIN, ImageFolder_down_img_TEST_many_fix_enhance_2
from compressai.zoo import models

from model import *
from utils.loss import *
from utils.metric import *
from utils.utils import *
from utils.distribution import *

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

        out1, out2, out_el1, out_el2, el1, el2 = wrap_model(img_base, img_el1, img_el2)

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
                    f"{i*len(d[0])}/{len(train_dataloader.dataset)}"
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

    image_save_path = '/raid/20204351/continuous_scalable_image_compression/saved_images_' + save_name
    os.makedirs(image_save_path, exist_ok=True)

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

        if epoch % 10 == 0:
            #output_img_save(img_base, epoch, count, scale_idx, image_save_path, '_div4_GT')
            output_img_save(img_el1, epoch, count, scale_idx, image_save_path, '_div2_GT')
            output_img_save(img_el2, epoch, count, scale_idx, image_save_path, '_div1_GT')

            #output_img_save(out_base['x_hat'], epoch, count, scale_idx, image_save_path, '_div4')
            output_img_save(out1, epoch, count, scale_idx, image_save_path, '_div2')
            output_img_save(out2, epoch, count, scale_idx, image_save_path, '_div1')

            output_img_save(el1, epoch, count, scale_idx, image_save_path, '_el2_in')
            output_img_save(out_el1['x_hat'], epoch, count, scale_idx, image_save_path, '_el2_out')
            output_img_save(el2, epoch, count, scale_idx, image_save_path, '_el1_in')
            output_img_save(out_el2['x_hat'], epoch, count, scale_idx, image_save_path, '_el1_out')

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.5f} |"
        f'\tMSE loss 1: {mse_loss_1.avg:.5f} |'
        f'\tMSE loss 2: {mse_loss_2.avg:.5f} |'
        f'\tBpp loss 1: {bpp_loss_1.avg:.5f} |'
        f'\tBpp loss 2: {bpp_loss_2.avg:.5f} |'
        f"\tAux loss(Res): {aux_loss_el.avg:.2f}\n"
    )

    testinfo = "Epoch: {} | Total(Test) loss:{} | Train loss:{} \n MSE loss 1:{}, MSE loss 2:{}\n"\
               "Bpp loss 1:{}, Bpp loss 2:{}\n, Aux loss(Res):{}, learning rate (comp):{}, learning rate (conti):{}\n"\
        .format(epoch, loss.avg, train_loss.avg, mse_loss_1.avg, mse_loss_2.avg,
                bpp_loss_1.avg, bpp_loss_2.avg, aux_loss_el.avg, learning_rate_comp, learning_rate_conti)

    now = time.localtime()
    testtime = "%04d/%02d/%02d %02d:%02d:%02d\n" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    with open('log_file_' + save_name + '.txt', 'a+') as f:
        f.write(testinfo)
        f.write(testtime)

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


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-mb",
        "--model_base",
        default="mbt2018-mean",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-mr",
        "--model_el",
        default="mbt2018-mean",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="examples/datasets_img",
        type=str,
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=300,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=5e-5,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.013,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=None,
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    # Option for Residual dense network (RDN)
    parser.add_argument('--G0', type=int, default=64,
                        help='default number of filters. (Use in RDN)')
    parser.add_argument('--RDNkSize', type=int, default=3,
                        help='default kernel size. (Use in RDN)')
    parser.add_argument('--RDNconfig', type=str, default='D',
                        help='parameters config of RDN. (Use in RDN)'
    )
    parser.add_argument(
        "--cuda", default=True, action="store_true", help="Use cuda"
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=0, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint_el", type=str, default="./examples/checkpoints/img_comp/checkpoint_conti_scalable_codec_v11(3stages_ori_loss_patch_512_mean_scale_bicubic_upsample_base_freeze_custom_round_layer_padding)_lambda_0.013_gradclip_1.0/epoch_0560.pth.tar", help="Path to a checkpoint")
    parser.add_argument("--checkpoint_prediction", type=str, default="/raid/20204351/conti_sr_JMSR/checkpoints/checkpoint_JMSR_cell_RDN_channels_64_RDNconfig_D_gradclip_1.0/epoch_0190.pth.tar", help="Path to a checkpoint")
    #parser.add_argument("--checkpoint", type=str, default="/raid/20204351/checkpoints/checkpoint_conti_scalable_codec_v12(3stages_ori_loss_patch_512_mean_scale_JMSR_cell_sigmoid_MLP_upsample_base_freeze_custom_round_layer_padding)_RDN_channels_64_RDNconfig_D_lambda_0.0035_gradclip_1.0/epoch_0160.pth.tar", help="Path to a checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--multi_gpu", default=True, help="multi gpu")
    parser.add_argument('--local_rank', type=int, default=0, help='Rank ID for processes')
    parser.add_argument(
        "--type", type=str, default="conti_scalable_codec_v12(3stages_ori_loss_patch_512_mean_scale_JMSR_cell_upsample_base_freeze_custom_round_layer_padding)", help="model_type"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    save_name = args.type + "_RDN_channels_" + str(args.G0) + "_RDNconfig_" + str(args.RDNconfig) + "_lambda_" + str(args.lmbda) + "_gradclip_" + str(args.clip_max_norm)

    log_dir = "./log_dir_eval2/" + save_name
    writer = SummaryWriter(log_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    synchronize()

    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder_down_img_TRAIN(args.dataset, split="train_512", transform=train_transforms)
    test_dataset = ImageFolder_down_img_TEST_many_fix_enhance_2(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    multi_gpu_sampler_train = torch.utils.data.DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=multi_gpu_sampler_train,
        shuffle=(multi_gpu_sampler_train is None),
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models[args.model_base](quality=4, pretrained=True, progress=True)
    net = net.to(device)
    net.update()

    net_enhance = models[args.model_el](quality=4, pretrained=True, progress=True)
    net_enhance = net_enhance.to(device)
    net_enhance.update()

    net_prediction = LIFF_prediction(args)
    net_prediction.to(device)

    if is_main_process():
        print("Lambda  : ", args.lmbda)
        print("The number of parameters (Base Layer) : ", count_parameters(net))
        print("The number of parameters (Enhance. Layer) : ", count_parameters(net_enhance))
        print("The number of parameters (Inter-Layer Arbirtrary Scale Prediction) : ", count_parameters(net_prediction))

    last_epoch = 0

    # load coarse model
    pretrained_el = torch.load(args.checkpoint_el, map_location=device)
    pretrained_prediction = torch.load(args.checkpoint_prediction, map_location=device)

    net_enhance.load_state_dict(pretrained_el["residual_state_dict"])
    net_prediction.load_state_dict(pretrained_prediction["model_state_dict"])

    net_enhance.update()

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net_enhance.load_state_dict(checkpoint["residual_state_dict"])
        net_prediction.load_state_dict(checkpoint["conti_sr_state_dict"])

    wrap_model = torch.nn.parallel.DistributedDataParallel(COMPASS(net, net_enhance, net_prediction, args).to(device),
                                                           device_ids=[args.local_rank],
                                                           find_unused_parameters=True)

    optimizer_el, aux_optimizer_el, optimizer_prediction = wrap_model.module.optimizer()

    if args.checkpoint:
        if is_main_process():
            print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1

        optimizer_el.load_state_dict(checkpoint["optimizer_res"])
        aux_optimizer_el.load_state_dict(checkpoint["aux_optimizer_res"])
        optimizer_prediction.load_state_dict(checkpoint["optimizer_conti_sr"])

    lr_scheduler_el = optim.lr_scheduler.StepLR(optimizer_el, step_size=100, gamma=0.5)
    lr_scheduler_prediction = optim.lr_scheduler.StepLR(optimizer_prediction, step_size=100, gamma=0.5)

    criterion = RateDistortionLoss(lmbda=args.lmbda)

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
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
            args.clip_max_norm,
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

            if args.save:
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
                    save_name,
                    epoch
                )


if __name__ == "__main__":
    main(sys.argv[1:])
