# 搭建神经网络
import os
import numpy as np
import torch
from torch import nn



class EncoderLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, 2, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class DecoderLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch * 2, 3, 1, 1),
            nn.BatchNorm1d(out_ch * 2),
            nn.ReLU(),
            nn.Conv1d(out_ch * 2, out_ch * 2, 3, 1, 1),
            nn.BatchNorm1d(out_ch * 2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(out_ch * 2, out_ch, 3, 2, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out, device='cuda'):
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        if (x_out.shape[2] > out.shape[2]):
            tmp = torch.zeros([1, x_out.shape[1], x_out.shape[2] - out.shape[2]]).to(device)
            out = torch.cat((out, tmp), dim=2)
        elif (x_out.shape[2] < out.shape[2]):
            tmp = torch.zeros([1, x_out.shape[1], out.shape[2] - x_out.shape[2]]).to(device)
            x_out = torch.cat((x_out, tmp), dim=2)

        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out


class HealthDetectorAug(nn.Module):
    def __init__(self, uwb_len, bre_len, hrt_len):
        super(HealthDetectorAug, self).__init__()
        self.uwb_len = uwb_len
        self.bre_len = bre_len-2
        self.hrt_len = hrt_len-2

        out_channels = [2 ** (i + 3) for i in range(8)]

        # Encoder:
        self.e1_1 = EncoderLayer(in_ch=18, out_ch=out_channels[0])  # 2->8
        self.e2_1 = EncoderLayer(in_ch=out_channels[0], out_ch=out_channels[1])  # 8->16
        self.e3_1 = EncoderLayer(in_ch=out_channels[1], out_ch=out_channels[2])  # 16->32
        self.e4_1 = EncoderLayer(in_ch=out_channels[2], out_ch=out_channels[3])  # 32->64
        self.e5_1 = EncoderLayer(in_ch=out_channels[3], out_ch=out_channels[4])  # 64->128
        self.e6_1 = EncoderLayer(in_ch=out_channels[4], out_ch=out_channels[5])  # 128->256

        self.e1_2 = EncoderLayer(in_ch=18, out_ch=out_channels[0])  # 2->8
        self.e2_2 = EncoderLayer(in_ch=out_channels[0], out_ch=out_channels[1])  # 8->16
        self.e3_2 = EncoderLayer(in_ch=out_channels[1], out_ch=out_channels[2])  # 16->32
        self.e4_2 = EncoderLayer(in_ch=out_channels[2], out_ch=out_channels[3])  # 32->64
        self.e5_2 = EncoderLayer(in_ch=out_channels[3], out_ch=out_channels[4])  # 64->128
        self.e6_2 = EncoderLayer(in_ch=out_channels[4], out_ch=out_channels[5])  # 128->256

        # Decoder:
        self.d1 = DecoderLayer(out_channels[6], out_channels[6])  # 512->256(2*256)
        self.d2 = DecoderLayer(out_channels[7], out_channels[5])  # 1024->256(2*512)
        self.d3 = DecoderLayer(out_channels[6], out_channels[4])  # 512->128(2*256)
        self.d4 = DecoderLayer(out_channels[5], out_channels[3])  # 256->64(2*128)
        self.d5 = DecoderLayer(out_channels[4], out_channels[2])  # 128->32(2*64)
        self.d6 = DecoderLayer(out_channels[3], out_channels[1])  # 256->64(2*128)

        # output
        self.trans = nn.Sequential(
            nn.Conv1d(2 * out_channels[1], out_channels[1], 3, 1, 1),
            nn.BatchNorm1d(out_channels[1]),
            nn.ReLU(),
            nn.Conv1d(out_channels[1], out_channels[0], 3, 1, 1),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Conv1d(out_channels[0], 1, 3, 1),
            nn.Sigmoid(),
        )

        self.bre_linear = nn.Sequential(
            nn.Linear(1478, 2956, bias=False),
            nn.ReLU(),
            nn.Linear(2956, 1478, bias=False),
            nn.ReLU(),
            nn.Linear(1478, self.bre_len, bias=False),
            nn.Sigmoid(),
        )

        self.hrt_linear = nn.Sequential(
            nn.Linear(1478, 2956, bias=False),
            nn.LeakyReLU(),
            nn.Linear(2956, 1478, bias=False),
            nn.LeakyReLU(),
            nn.Linear(1478, self.hrt_len, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, uwb_1, uwb_2):
        out_1_1, out1_1 = self.e1_1(uwb_1)  # 8,8
        out_2_1, out2_1 = self.e2_1(out1_1)  # 16,16
        out_3_1, out3_1 = self.e3_1(out2_1)  # 32,32
        out_4_1, out4_1 = self.e4_1(out3_1)  # 64,64
        out_5_1, out5_1 = self.e5_1(out4_1)  # 128,128
        out_6_1, out6_1 = self.e6_1(out5_1)  # 256,256

        out_1_2, out1_2 = self.e1_2(uwb_2)  # 8,8
        out_2_2, out2_2 = self.e2_2(out1_2)  # 16,16
        out_3_2, out3_2 = self.e3_2(out2_2)  # 32,32
        out_4_2, out4_2 = self.e4_2(out3_2)  # 64,64
        out_5_2, out5_2 = self.e5_2(out4_2)  # 128,128
        out_6_2, out6_2 = self.e6_2(out5_2)  # 256,256

        out6 = torch.cat((out6_1, out6_2), dim=1)  # 256+256=512
        out_6 = torch.cat((out_6_1, out_6_2), dim=1)  # 256+256=512
        out_m6 = self.d1(out6, out_6)  # 512->1024

        out_5 = torch.cat((out_5_1, out_5_2), dim=1)  # 128+128=256
        out_m5 = self.d2(out_m6, out_5)  # 1024->256->512

        out_4 = torch.cat((out_4_1, out_4_2), dim=1)
        out_m4 = self.d3(out_m5, out_4)

        out_3 = torch.cat((out_3_1, out_3_2), dim=1)
        out_m3 = self.d4(out_m4, out_3)

        out_2 = torch.cat((out_2_1, out_2_2), dim=1)
        out_m2 = self.d5(out_m3, out_2)

        out_1 = torch.cat((out_1_1, out_1_2), dim=1)
        out_m1 = self.d6(out_m2, out_1)

        out = self.trans(out_m1)
        out_bre = self.bre_linear(out)

        return out_bre, out6_1, out6_2


if __name__ == '__main__':
    sig_len = 37*40
    HealthDetect = HealthDetectorAug(sig_len)
    input = torch.ones((1, 3, sig_len))
    output = HealthDetect(input)
