#!/usr/bin/python3
# coding=utf-8

from cbam import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from Res2Net_v1b import res2net50_v1b_26w_4s
from pvtv2 import pvt_v2_b2
from einops import rearrange
import numbers
from typing import List
from timm.models.layers import DropPath, trunc_normal_
from functools import partial

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential,nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, (nn.ReLU,nn.Sigmoid,nn.SiLU,nn.GELU,nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d,nn.Conv1d,nn.Upsample,nn.MaxPool2d,nn.Identity,nn.Dropout,nn.ZeroPad2d)):
            pass
        else:
            m.initialize()

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

    def initialize(self):
        weight_init(self)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def initialize(self):
        weight_init(self)

class Grafting1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(64)
        self.lny = nn.LayerNorm(64)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = Mlp(64, 64*4)

    # x:B,64,H,W
    # y:B,64,H,W
    def forward(self, x, y):
        batch_size, chanel , H , W = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        sc = x  # B,C,H,W
        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)  # B,HW,C
        sc1 = x  # B,HW,C
        x = self.lnx(x)
        y = y.view(batch_size, chanel, -1).permute(0, 2, 1)  # B,HW,C
        y = self.lny(y)

        B, N, C = x.shape
        y_k = self.k(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                      4)  # 1,B,heads,HW,C/heads
        x_qv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                        4)  # 2,B,heads,HW,C/heads
        x_q, x_v = x_qv[0], x_qv[1]  # B,heads,HW,C/heads
        y_k = y_k[0]  # B,heads,HW,C/heads
        attn_bmm = (x_q @ y_k.transpose(-2, -1)) * self.scale  # B,heads,HW,HW
        attn = attn_bmm.softmax(dim=-1)

        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C)  # B,HW,C

        x = self.proj(x)  # B,HW,C
        x = (x + sc1)


        x = self.conv2(x,H,W) + x
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, chanel, *sc.size()[2:])  # B,C,H,W

        return x

    def initialize(self):
        weight_init(self)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

    def initialize(self):
        weight_init(self)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

    def initialize(self):
        weight_init(self)

class merge(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cross_attention = Grafting1(64)
        self.conv3 = BaseConv2d(in_dim*2, in_dim)
        self.conv2 = BaseConv2d(in_dim*2, in_dim)

        self.channalatten = ChannelAttention(64)
        self.spatialatten = SpatialAttention()

    def forward(self, m_trans_feats,l_trans_feats):
        s1, s2, s3, s4 = m_trans_feats[0],m_trans_feats[1],m_trans_feats[2],m_trans_feats[3]
        r1, r2, r3, r4 = l_trans_feats[0],l_trans_feats[1],l_trans_feats[2],l_trans_feats[3]

        out1 = r1

        out2,_ = torch.stack((r2,s1),dim=1).max(1)

        out3, _ = torch.stack((r3, s2),dim=1).max(1)

        out4 = self.cross_attention(s3, r4)


        out1 = out1.mul(self.channalatten(out1))
        out1 = out1.mul(self.spatialatten(out1))
        out2 = out2.mul(self.channalatten(out2))
        out2 = out2.mul(self.spatialatten(out2))
        out3 = out3.mul(self.channalatten(out3))
        out3 = out3.mul(self.spatialatten(out3))
        out4 = out4.mul(self.channalatten(out4))
        out4 = out4.mul(self.spatialatten(out4))

        return out1,out2,out3,out4

    def initialize(self):
        weight_init(self)

class refine(nn.Module):

    def __init__(self, in_chan, out_chan=64):
        super(refine, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        return feat
    def initialize(self):
        weight_init(self)

class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.att = nn.Sequential(
            nn.Conv2d(64*2, 64, 3, bias=False, padding=1),
            nn.Conv2d(64, 1, 3, bias=False, padding=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )

    def forward(self, x):
        block1 = self.att(x)
        return block1

    def initialize(self):
        weight_init(self)

class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=True) -> None:
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basicconv(x)

    def initialize(self):
        weight_init(self)

class fusion(nn.Module):

    def __init__(self) -> None:
        """
        Args:
            channels: It should a list which denotes the same channels
                      of encoder side outputs(skip connection features).
        """
        super(fusion, self).__init__()
        # decoder layer 5
        self.conv5 = nn.Sequential(
            BaseConv2d(64, 64),
            BaseConv2d(64, 64),
            BaseConv2d(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # decoder layer 4
        self.conv4 = nn.Sequential(
            BaseConv2d(64 * 2, 64),
            BaseConv2d(64, 64),
            BaseConv2d(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # decoder layer 3
        self.conv3 = nn.Sequential(
            BaseConv2d(64 * 2, 64),
            BaseConv2d(64, 64),
            BaseConv2d(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # decoder layer 2
        self.conv2 = nn.Sequential(
            BaseConv2d(64 * 2, 64),
            BaseConv2d(64,64),
        )


        self.c1 = nn.Sequential(
            BaseConv2d(64, 64, kernel_size=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            BaseConv2d(64 + 64, 64, kernel_size=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.c3 = nn.Sequential(
            BaseConv2d(64 + 64 + 64, 64,kernel_size=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        # decoder out
        self.conv_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, decoder_list):
        assert len(decoder_list) == 4

        # decoder layer 5
        decoder_map5 = self.conv5(decoder_list[0])

        # decoder layer 4
        semantic_block4 = self.c1(F.interpolate(decoder_list[0], scale_factor=2))
        assert semantic_block4.size() == decoder_list[1].size()
        short4 = torch.mul(semantic_block4, decoder_list[1]) + decoder_list[1]
        decoder_map4_input = torch.cat([decoder_map5, short4], dim=1)
        decoder_map4 = self.conv4(decoder_map4_input)

        # decoder layer 3
        semantic_block3 = self.c2(
            torch.cat([F.interpolate(decoder_list[0], scale_factor=4),
                       F.interpolate(decoder_list[1], scale_factor=2)], dim=1))
        assert semantic_block3.size() == decoder_list[2].size()
        short3 = torch.mul(semantic_block3, decoder_list[2]) + decoder_list[2]
        decoder_map3_input = torch.cat([decoder_map4, short3], dim=1)
        decoder_map3 = self.conv3(decoder_map3_input)

        # decoder layer 2
        semantic_block2 = self.c3(
            torch.cat([F.interpolate(decoder_list[0], scale_factor=8),
                       F.interpolate(decoder_list[1], scale_factor=4),
                       F.interpolate(decoder_list[2], scale_factor=2)], dim=1))
        assert semantic_block2.size() == decoder_list[3].size()
        short2 = torch.mul(semantic_block2, decoder_list[3]) + decoder_list[3]
        decoder_map2_input = torch.cat([decoder_map3, short2], dim=1)
        decoder_map2 = self.conv2(decoder_map2_input)


        return decoder_map2

    def initialize(self):
        weight_init(self)

class TransLayer(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.c4_down = nn.Sequential(BaseConv2d(2048, out_c))
        self.c3_down = nn.Sequential(BaseConv2d(1024, out_c))
        self.c2_down = nn.Sequential(BaseConv2d(512, out_c))
        self.c1_down = nn.Sequential(BaseConv2d(256, out_c))

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 4
        c1, c2, c3, c4= xs
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c1,c2,c3,c4

    def initialize(self):
        weight_init(self)
class TransLayer2(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.c4_down = nn.Sequential(BaseConv2d(512, out_c))
        self.c3_down = nn.Sequential(BaseConv2d(320, out_c))
        self.c2_down = nn.Sequential(BaseConv2d(128, out_c))
        self.c1_down = nn.Sequential(BaseConv2d(64, out_c))

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 4
        c1, c2, c3, c4= xs
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c1,c2,c3,c4

    def initialize(self):
        weight_init(self)


class FDNet(nn.Module):
    def __init__(self, cfg, pretrain=False):
        super(FDNet, self).__init__()
        self.cfg = cfg
        self.bkbone = res2net50_v1b_26w_4s(pretrained=pretrain)
        self.bkbone2 = pvt_v2_b2()  # [64, 128, 320, 512]
        save_model = torch.load('./out/pvt_v2_b2.pth')
        model_dict = self.bkbone2.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.bkbone2.load_state_dict(model_dict)
        self.translayer = TransLayer(out_c=64)  # [c4, c3, c2, c1] channel-->64
        self.translayer2 = TransLayer2(out_c=64)
        self.merge_layers = merge(64)
        self.fusion = fusion()

        self.fn = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace = True)
        )
        self.fp = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace = True)
        )

        self.refine1 = refine(128)
        self.attention1 = attention()

        self.linear_fn = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear_fp = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear_p = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def encoder_translayer(self, x):
        en_feats = self.bkbone(x)
        trans_feats = self.translayer(en_feats)
        return trans_feats

    def encoder_translayer2(self, x):
        en_feats = self.bkbone2(x)
        trans_feats = self.translayer2(en_feats)
        return trans_feats

    def forward(self, image, shape=None):

        x, x_l = image[0], image[1]
        m_trans_feats = self.encoder_translayer2(x)  # [72,72,64],[36,36,64],[18,24,64],[18,12,24]
        l_trans_feats = self.encoder_translayer(x_l)  # [144,144,64],[72,72,64],[36,36,64],[18,24,64]

        out1,out2,out3,out4 = self.merge_layers(m_trans_feats,l_trans_feats)
        # [B,64,144,144]    [B,64,72,72]    [B,64,36,36]    [B,64,18,18]
        datalist = []
        datalist.append(out4)
        datalist.append(out3)
        datalist.append(out2)
        datalist.append(out1)
        predict = self.fusion(datalist)    #[B,64,144,144]


        fn = self.fn(predict)
        fp = self.fp(predict)
        predict_feature = predict
        predict_feature = (1 + self.attention1(torch.cat((predict_feature, fn), 1))) * predict_feature
        predict_feature = predict_feature - self.refine1(torch.cat((predict_feature,fp),1))

        fn = self.linear_fn(fn)
        fp = self.linear_fp(fp)
        predict_feature = self.linear_p(predict_feature)


        shape = x_l.size()[2:]
        fn = F.interpolate(fn, size=shape, mode='bilinear')
        fp = F.interpolate(fp, size=shape, mode='bilinear')
        predict_feature = F.interpolate(predict_feature, size=shape, mode='bilinear')

        return fn,fp,predict_feature

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)





