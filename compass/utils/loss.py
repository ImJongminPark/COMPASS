import math

import torch
import torch.nn as nn


class RateDistortionLossForLayer(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output_bpp, out_x_hat, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        # R loss
        out_bpp = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output_bpp.values()
        )

        # D loss
        out_mse = self.mse(out_x_hat, target)

        return out_bpp, out_mse


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.lmbda = lmbda
        self.RD_loss = RateDistortionLossForLayer()

    def forward(self, out_res1, out_res2, out1, out2, img_res1, img_res2):
        out = {}

        out["bpp_loss_1"], out["mse_loss_1"] = self.RD_loss(out_res1["likelihoods"], out1, img_res1)
        out["bpp_loss_2"], out["mse_loss_2"] = self.RD_loss(out_res2["likelihoods"], out2, img_res2)

        bpp_total = out["bpp_loss_1"] + out["bpp_loss_2"]
        mse_total = 255 ** 2 * self.lmbda * (out["mse_loss_1"] + out["mse_loss_2"])

        out["loss"] = bpp_total + mse_total

        return out
