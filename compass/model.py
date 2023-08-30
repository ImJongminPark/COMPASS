import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from compass.utils.metric import *


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


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class LIFF_prediction(nn.Module):
    def __init__(self, cfg):
        super(LIFF_prediction, self).__init__()
        G0 = cfg['LIFF']['G0']
        kSize = cfg['LIFF']['RDNkSize']

        self.kernel_size = 3
        self.inC = G0
        self.outC = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (4, 8, 64),
            'D': (4, 4, 32),
            'E': (4, 4, 16)
        }[cfg['LIFF']['RDNconfig']]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Residual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.imnet = nn.Sequential(*[
            nn.Linear(G0 * 9 + 2 + 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.inC * self.outC * self.kernel_size * self.kernel_size),
        ])

    def forward(self, x, img_hr_coord, img_hr_cell):
        # ------------------------------------- Feature Learning Module ------------------------------------- #
        f_1 = self.SFENet1(x)
        x = self.SFENet2(f_1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f_1
        # -------------------------------------------------------------------------------------------------- #

        x = F.unfold(x, 3, padding=1).view(x.shape[0], x.shape[1] * 9, x.shape[2], x.shape[3])

        x_coord = make_coord(x.shape[-2:], flatten=False).cuda()
        x_coord = x_coord.permute(2, 0, 1).contiguous().unsqueeze(0)
        x_coord = x_coord.expand(x.shape[0], 2, *x.shape[-2:])

        img_hr_coord_ = img_hr_coord.clone()
        img_hr_coord_ = img_hr_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        img_hr_coord_ = img_hr_coord_.permute(0, 2, 3, 1).contiguous()
        img_hr_coord_ = img_hr_coord_.view(img_hr_coord.size(0), -1, img_hr_coord.size(1))

        q_feat = F.grid_sample(x, img_hr_coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
        q_feat = q_feat.view(img_hr_coord.size(0), -1, img_hr_coord.size(2)*img_hr_coord.size(3)).permute(0, 2, 1).contiguous()  

        q_coord = F.grid_sample(x_coord, img_hr_coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
        q_coord = q_coord.view(img_hr_coord.size(0), -1, img_hr_coord.size(2) * img_hr_coord.size(3)).permute(0, 2, 1).contiguous()

        rel_coord = img_hr_coord_ - q_coord
        rel_coord[:, :, 0] *= x.shape[-2]
        rel_coord[:, :, 1] *= x.shape[-1]
        inp = torch.cat([q_feat, rel_coord], dim=-1)

        img_hr_cell_ = img_hr_cell.clone()
        img_hr_cell_ = img_hr_cell_.permute(0, 2, 3, 1).contiguous()
        rel_cell = img_hr_cell_.view(img_hr_cell.size(0), -1, img_hr_cell.size(1))
        rel_cell[:, :, 0] *= x.shape[-2]
        rel_cell[:, :, 1] *= x.shape[-1]

        inp = torch.cat([inp, rel_cell], dim=-1)

        local_weight = self.imnet(inp)
        local_weight = local_weight.view(x.size(0), -1, x.size(1), 3)

        cols = q_feat.unsqueeze(2)

        out = torch.matmul(cols, local_weight).squeeze(2).permute(0, 2, 1).contiguous().view(img_hr_coord.size(0), -1, img_hr_coord.size(2), img_hr_coord.size(3))

        return out


class COMPASS(nn.Module):
    def __init__(self, model, model_el, model_prediction, cfg):
        super(COMPASS, self).__init__()
        self.model = model
        self.model_el = model_el
        self.model_prediction = model_prediction
        self.cfg = cfg

    def configure_optimizers(self, net, cfg):
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

    def optimizer(self):
        optimizer_el, aux_optimizer_el = self.configure_optimizers(self.model_el, self.cfg)
        optimizer_prediction = optim.Adam(self.model_prediction.parameters(), lr=self.cfg['lr'], betas=(0.9, 0.999))

        return optimizer_el, aux_optimizer_el, optimizer_prediction

    def get_local_grid(self, img):
        local_grid = make_coord(img.shape[-2:], flatten=False).cuda()
        local_grid = local_grid.permute(2, 0, 1).unsqueeze(0)
        local_grid = local_grid.expand(img.shape[0], 2, *img.shape[-2:])

        return local_grid

    def get_cell(self, img, local_grid):
        cell = torch.ones_like(local_grid)
        cell[:, 0] *= 2 / img.size(2)
        cell[:, 1] *= 2 / img.size(3)

        return cell
    
    def inference(self, img_base, img_el1, img_el2):
        local_grid_el1 = self.get_local_grid(img_el1)
        cell_el1 = self.get_cell(img_el1, local_grid_el1)

        local_grid_el2 = self.get_local_grid(img_el2)
        cell_el2 = self.get_cell(img_el1, local_grid_el2)

        # Image Compression/Reconstruction
        out_enc_base = self.model.compress(img_base)
        out_dec_base = self.model.decompress(out_enc_base["strings"], out_enc_base["shape"])
        out_comp_base = out_dec_base['x_hat']
        bit_base = sum(len(s[0]) for s in out_enc_base["strings"]) * 8.0

        pred1 = self.model_prediction(out_dec_base['x_hat'], local_grid_el1, cell_el1)
        el1 = img_el1 - pred1
        out_enc_el1 = self.model_el.compress(el1)
        out_dec_el1 = self.model_el.decompress(out_enc_el1["strings"], out_enc_el1["shape"])
        out_comp_el1 = out_dec_el1['x_hat'] + pred1
        bit_el1 = bit_base + sum(len(s[0]) for s in out_enc_el1["strings"]) * 8.0

        pred2 = self.model_prediction(out_comp_el1, local_grid_el2, cell_el2)
        el2 = img_el2 - pred2
        out_enc_el2 = self.model_el.compress(el2)
        out_dec_el2 = self.model_el.decompress(out_enc_el2["strings"], out_enc_el2["shape"])
        out_comp_el2 = out_dec_el2['x_hat'] + pred2
        bit_el2 = bit_el1 + sum(len(s[0]) for s in out_enc_el2["strings"]) * 8.0

        psnr_base = psnr(img_base, out_comp_base.clamp_(0, 1))
        psnr_el1 = psnr(img_el1, out_comp_el1.clamp_(0, 1))
        psnr_el2 = psnr(img_el2, out_comp_el2.clamp_(0, 1))

        return psnr_base, psnr_el1, psnr_el2, bit_base, bit_el1, bit_el2

    def forward(self, img_base, img_el1, img_el2):
        local_grid_el1 = self.get_local_grid(img_el1)
        cell_el1 = self.get_cell(img_el1, local_grid_el1)

        local_grid_el2 = self.get_local_grid(img_el2)
        cell_el2 = self.get_cell(img_el1, local_grid_el2)

        out_base = self.model(img_base)

        pred1 = self.model_prediction(out_base['x_hat'], local_grid_el1, cell_el1)
        el1 = img_el1 - pred1
        out_el1 = self.model_el(el1)
        out1 = out_el1['x_hat'] + pred1

        pred2 = self.model_prediction(out1, local_grid_el2, cell_el2)
        el2 = img_el2 - pred2
        out_el2 = self.model_el(el2)
        out2 = out_el2['x_hat'] + pred2

        return out1, out2, out_el1, out_el2, el1, el2

