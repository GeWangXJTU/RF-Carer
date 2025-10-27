# 搭建神经网络
import os
import numpy as np
import torch
from torch import nn


# DroppingDetector处理的是一个二分类问题
# 将Heatmap作为输入，输出每个像素是否有Dropping信息的mask
# 数据集和Pascal voc 2012数据集有较高的相似性（简单很多）
# 可以参考CV领域的图像分割方法
# 拟参考经典网络U-Net的结构实现功能，参考链接：https://blog.csdn.net/kobayashi_/article/details/108951993

# 当前将问题定义为：输入一个三通道的一维tensor（3*1*80），输出单通道的一维mask（1*1*80）
# Loss函数采用nn.BCELoss()

class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.dif_ch = out_channel - in_channel

        self.conv1 = nn.Conv1d(self.in_ch, self.dif_ch, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(self.dif_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(self.dif_ch, self.dif_ch, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(self.dif_ch)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = nn.MaxPool1d(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.in_ch != self.out_ch:
            out = torch.cat((x, out), dim=1)
        else:
            out += x

        out = self.relu(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderLayer, self).__init__()
        self.basic = BasicBlock(in_ch, out_ch)
        self.downsample = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, 2, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.basic(x)
        out_2 = self.downsample(out)
        return out, out_2


class DecoderLayer(nn.Module):
    def __init__(self, in_ch, out_ch, device):
        super(DecoderLayer, self).__init__()
        self.device = device

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

    def forward(self, x, out):
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        if (x_out.shape[2] > out.shape[2]):
            tmp = torch.zeros([1, x_out.shape[1], x_out.shape[2] - out.shape[2]]).to(self.device)
            out = torch.cat((out, tmp), dim=2)
        elif (x_out.shape[2] < out.shape[2]):
            tmp = torch.zeros([1, x_out.shape[1], out.shape[2] - x_out.shape[2]]).to(self.device)
            x_out = torch.cat((x_out, tmp), dim=2)

        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out


# 搭建神经网络
class HealthDetector(nn.Module):
    def __init__(self, uwb_len, bre_len, hrt_len, device):
        super(HealthDetector, self).__init__()
        self.uwb_len = uwb_len
        self.bre_len = bre_len - 2
        self.hrt_len = hrt_len - 2
        self.device = device
        out_channels = [2 ** (i + 3) for i in range(8)]

        # Encoder:
        self.e1_1 = EncoderLayer(in_ch=3, out_ch=out_channels[0])  # 3->8
        self.e2_1 = EncoderLayer(in_ch=out_channels[0], out_ch=out_channels[1])  # 8->16
        self.e3_1 = EncoderLayer(in_ch=out_channels[1], out_ch=out_channels[2])  # 16->32
        self.e4_1 = EncoderLayer(in_ch=out_channels[2], out_ch=out_channels[3])  # 32->64
        self.e5_1 = EncoderLayer(in_ch=out_channels[3], out_ch=out_channels[4])  # 64->128
        # self.dropout_1 = nn.Dropout(0.3)
        self.e6_1 = EncoderLayer(in_ch=out_channels[4], out_ch=out_channels[5])  # 128->256

        self.e1_2 = EncoderLayer(in_ch=3, out_ch=out_channels[0])  # 3->8
        self.e2_2 = EncoderLayer(in_ch=out_channels[0], out_ch=out_channels[1])  # 8->16
        self.e3_2 = EncoderLayer(in_ch=out_channels[1], out_ch=out_channels[2])  # 16->32
        self.e4_2 = EncoderLayer(in_ch=out_channels[2], out_ch=out_channels[3])  # 32->64
        self.e5_2 = EncoderLayer(in_ch=out_channels[3], out_ch=out_channels[4])  # 64->128
        # self.dropout_2 = nn.Dropout(0.3)
        self.e6_2 = EncoderLayer(in_ch=out_channels[4], out_ch=out_channels[5])  # 128->256

        # Decoder:
        self.d1 = DecoderLayer(out_channels[6], out_channels[6], self.device)  # 512->256(2*256)
        self.d2 = DecoderLayer(out_channels[7], out_channels[5], self.device)  # 1024->256(2*512)
        self.d3 = DecoderLayer(out_channels[6], out_channels[4], self.device)  # 512->128(2*256)
        self.d4 = DecoderLayer(out_channels[5], out_channels[3], self.device)  # 256->64(2*128)
        # self.dropout_d = nn.Dropout(0.3)
        self.d5 = DecoderLayer(out_channels[4], out_channels[2], self.device)  # 128->32(2*64)
        self.d6 = DecoderLayer(out_channels[3], out_channels[1], self.device)  # 256->64(2*128)

        # output
        self.trans = nn.Sequential(
            nn.Conv1d(2 * out_channels[1], out_channels[0], 3, 1, 1),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Conv1d(out_channels[0], 1, 3, 1),
            nn.Sigmoid(),
        )

        self.bre_linear = nn.Sequential(
            nn.Linear(1478, self.bre_len, bias=False),
            nn.Sigmoid(),
        )

        # self.hrt_linear = nn.Sequential(
        #     nn.Linear(1478, 2956, bias=False),
        #     nn.LeakyReLU(),
        #     nn.Linear(2956, 1478, bias=False),
        #     nn.LeakyReLU(),
        #     nn.Linear(1478, self.hrt_len, bias=False),
        #     nn.Sigmoid(),
        # )

        self.hrt_linear_1 = nn.Sequential(
            nn.Linear(1478*2, 1478, bias=False),
            nn.ReLU(),
        )
        self.hidden_size = 20
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=20, num_layers=1,
                                  bias=False, batch_first=True, bidirectional=True)

        self.hrt_linear_2 = nn.Sequential(
            nn.Linear(self.hidden_size*2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, uwb_1, uwb_2):
        out_1_1, out1_1 = self.e1_1(uwb_1)  # 8,8
        out_2_1, out2_1 = self.e2_1(out1_1)  # 16,16
        out_3_1, out3_1 = self.e3_1(out2_1)  # 32,32
        out_4_1, out4_1 = self.e4_1(out3_1)  # 64,64
        out_5_1, out5_1 = self.e5_1(out4_1)  # 128,128
        # out5_1 = self.dropout_1(out5_1)
        out_6_1, out6_1 = self.e6_1(out5_1)  # 256,256

        out_1_2, out1_2 = self.e1_2(uwb_2)  # 8,8
        out_2_2, out2_2 = self.e2_2(out1_2)  # 16,16
        out_3_2, out3_2 = self.e3_2(out2_2)  # 32,32
        out_4_2, out4_2 = self.e4_2(out3_2)  # 64,64
        out_5_2, out5_2 = self.e5_2(out4_2)  # 128,128
        # out5_2 = self.dropout_2(out5_2)
        out_6_2, out6_2 = self.e6_2(out5_2)  # 256,256

        out6 = torch.cat((out6_1, out6_2), dim=1)  # 256+256=512
        out_6 = torch.cat((out_6_1, out_6_2), dim=1)  # 256+256=512
        out_m6 = self.d1(out6, out_6)  # 512->1024

        out_5 = torch.cat((out_5_1, out_5_2), dim=1)  # 128+128=256
        out_m5 = self.d2(out_m6, out_5)  # 1024->256->512

        out_4 = torch.cat((out_4_1, out_4_2), dim=1)
        out_m4 = self.d3(out_m5, out_4).to(self.device)

        out_3 = torch.cat((out_3_1, out_3_2), dim=1)
        out_m3 = self.d4(out_m4, out_3)
        # out_m3 = self.dropout_d(out_m3)

        out_2 = torch.cat((out_2_1, out_2_2), dim=1)
        out_m2 = self.d5(out_m3, out_2)

        out_1 = torch.cat((out_1_1, out_1_2), dim=1)
        out_m1 = self.d6(out_m2, out_1)

        out = self.trans(out_m1)
        out_bre = self.bre_linear(out)
        # out_hrt = self.hrt_linear(out)

        # out_combine = torch.cat((out, out_bre), dim=2)
        # out_brt_1 = self.hrt_linear_1(out_combine)
        # out_brt_1 = out_brt_1.permute(0, 2, 1)
        # out_brt_2, _ = self.lstm_layer(out_brt_1)
        # out_brt_2 = torch.squeeze(out_brt_2)
        # out_hrt = self.hrt_linear_2(out_brt_2)
        # out_hrt = torch.squeeze(out_hrt)
        return out_bre, out6_1, out6_2


if __name__ == '__main__':
    sig_len = 37 * 40
    HealthDetect = HealthDetector(sig_len)
    input = torch.ones((1, 3, sig_len))
    output = HealthDetect(input)
