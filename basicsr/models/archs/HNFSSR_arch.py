# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet

@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.NAFNet_arch import LayerNorm2d, NAFBlock
from basicsr.models.archs.arch_util import MySequential
from basicsr.models.archs.local_arch import Local_Base


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class DAC(nn.Module):
    def __init__(self, n_channels):
        super(DAC, self).__init__()

        self.mean = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )
        self.std = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )

    def forward(self, observed_feat, referred_feat):
        assert (observed_feat.size()[:2] == referred_feat.size()[:2])
        size = observed_feat.size()
        referred_mean, referred_std = calc_mean_std(referred_feat)
        observed_mean, observed_std = calc_mean_std(observed_feat)

        normalized_feat = (observed_feat - observed_mean.expand(
            size)) / observed_std.expand(size)
        referred_mean = self.mean(referred_mean)
        referred_std = self.std(referred_std)
        output = normalized_feat * referred_std.expand(size) + referred_mean.expand(size)
        return output

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x + factor * (new_x - x) for x, new_x in zip(feats, new_feats)])
        return new_feats

class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        return hessian

class DiEnDec(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super(DiEnDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.gate(self.decoder(self.encoder(x)))
        return output

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats

class HESSBlock(nn.Module):
    def __init__(self,n_channels,act):
        super().__init__()
        self.coder = nn.Sequential(DiEnDec(3, act))
        self.dac = nn.Sequential(DAC(n_channels))
        self.hessian3 = nn.Sequential(MSHF(n_channels, kernel=3))
        self.hessian5 = nn.Sequential(MSHF(n_channels, kernel=5))
        self.hessian7 = nn.Sequential(MSHF(n_channels, kernel=7))

    def foward(self,x):
        sz = x.size()
        feat_0 = x
        hessian3 = self.hessian3(feat_0)
        hessian5 = self.hessian5(feat_0)
        hessian7 = self.hessian7(feat_0)
        hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                             torch.mean(hessian5, dim=1, keepdim=True),
                             torch.mean(hessian7, dim=1, keepdim=True))
                            , 1)
        hessian = self.coder(hessian)
        attention = torch.sigmoid(self.dac[0](hessian.expand(sz), x))
        resout = x * attention
        feat_out = x + resout
        return feat_out

# class HNANet(nn.Module):
#     def __init__(self,c, n_channels, n_blocks, act, attention, drop_out_rate=0.):
#         super().__init__()
#         res_blocks = [NAFBlock(c,drop_out_rate = drop_out_rate) for _ in range(n_blocks)]
#         self.body_block = nn.Sequential(*res_blocks)
#         self.attention = attention
#         if attention:
#             self.coder = nn.Sequential(DiEnDec(3, act))
#             self.dac = nn.Sequential(DAC(n_channels))
#             self.hessian3 = nn.Sequential(MSHF(n_channels, kernel=3))
#             self.hessian5 = nn.Sequential(MSHF(n_channels, kernel=5))
#             self.hessian7 = nn.Sequential(MSHF(n_channels, kernel=7))
#
#     def forward(self, x):
#         sz = x.size()
#         resin = self.body_block(x)
#         # print('>>>>>>>>resin>>>>>>>>>>',x.shape)
#         if self.attention:
#             hessian3 = self.hessian3(resin)
#             hessian5 = self.hessian5(resin)
#             hessian7 = self.hessian7(resin)
#             hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
#                                  torch.mean(hessian5, dim=1, keepdim=True),
#                                  torch.mean(hessian7, dim=1, keepdim=True))
#                                 , 1)
#             hessian = self.coder(hessian)
#             attention = torch.sigmoid(self.dac[0](hessian.expand(sz), x))
#             resout = resin * attention
#         else:
#             resout = resin
#
#         output = resout + x
#
#         return output

class HNAFSRNet(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''

    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0.,act=nn.ReLU(True),
                 fusion_from=-1, fusion_to=-1, dual=False,n_channels=32,n_blocks=10):
        super().__init__()
        self.dual = dual  # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate,
                NAFBlockSR(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate,
                )) for i in range(num_blks)]

        )
        self.detail = HESSBlock(n_channels=width,act=act)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale ** 2, kernel_size=3, padding=1, stride=1,
                      groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
        self.block = n_blocks
    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp,)
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        l=[]
        for x in feats:
            for i in range(self.block):
                feat = self.detail.foward(x)
            l.append(feat)
        out = torch.cat([self.up(x) for x in l], dim=1)
        # print('>>>>>>>>>>>>>out>>>>>>>>>>>>>>>',out.shape)
        out = out + inp_hr
        return out

# class HNAFSRNet(nn.Module):
#     def __init__(self,up_scale=4, width=128, num_blks=128, img_channel=3, drop_path_rate=0.3, drop_out_rate=0.,act=nn.ReLU(inplace=True),
#                                fusion_from=-1, fusion_to=-1, dual=True,n_channels=32,n_blocks=10,attention=True):
#         super(HNAFSRNet, self).__init__()
#         self.nafssr = NAFNetSR(up_scale=4, width=128, num_blks=128, img_channel=3, drop_path_rate=0.3, drop_out_rate=0.,
#                                fusion_from=-1, fusion_to=-1, dual=dual)
#         self.hnafnet = HNANet(c=32,n_channels=32,n_blocks=32,attention=True,act=nn.ReLU(inplace=True),drop_out_rate=0.)
#
#         self.output = nn.Conv2d(in_channels = 6,out_channels = 32,kernel_size = 3,stride =1,padding=1,bias =True)
#         self.outend = nn.Conv2d(in_channels = 32,out_channels =6 , kernel_size =1)
#         self.up = nn.Sequential(nn.PiexlShuffle(up_scale))
#
#     def forward(self,x):
#         inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
#         x = self.nafssr.forward(x)
#         # print('>>>>>>>>>>nafssr>>>>>>>>>>>>>>',x.shape)
#         if x.is_cuda:
#             self.hnafnet.cuda()
#         else:
#             self.hnafnet.cpu()
#         x = self.output(x)
#         x = self.hnafnet.forward(x)
#         x = self.outend(x)
#         up = self.up(x)
#         out = inp_hr+up
#         # print('>>>>>>>>>>HNAFNET>>>>>>>>>',x.shape)
#         return out
    
class HNAFSSR(Local_Base, HNAFSRNet):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        HNAFSRNet.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    num_blks = 128
    width = 128
    droppath = 0.1
    train_size = (1, 6, 30, 90)

    # net = NAFSSR(up_scale=2, train_size=train_size, fast_imp=True, width=width, num_blks=num_blks,
    #              drop_path_rate=droppath)

    net = HNAFSRNet(up_scale=4, width=128, num_blks=128, img_channel=3, drop_path_rate=0.3, drop_out_rate=0.,act=nn.ReLU(True),
                               fusion_from=-1, fusion_to=1000, dual=True,n_channels=32)

    inp_shape = (6, 64, 64)

    img = torch.randn(1, 6, 256, 256)
    print('>input: ', img.shape)
    x = net.forward(img)
    print('>output: ', x.shape)
    # from ptflops import get_model_complexity_info
    #
    # FLOPS = 0
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    #
    # # params = float(params[:-4])
    # print(params)
    # macs = float(macs[:-4]) + FLOPS / 10 ** 9
    #
    # print('mac', macs, params)

    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))




