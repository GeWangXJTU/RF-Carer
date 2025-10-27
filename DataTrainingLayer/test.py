import os
import random
import torch.fft
import numpy as np
import torch.optim
from matplotlib import pyplot
from torch import nn
from torch.utils.data import DataLoader

from Dataset import HealthDataset


# from torch.utils.tensorboard import SummaryWriter
# from model import HealthDetector
# from ResModel import HealthDetector
from calRate import calRate


# Parameters:
total_train_step = 0
epoch = 50  # Number of round
learning_rate = 0.001
second = 37
fps_uwb = 40
fps_ecg = 125
fps_bre = 50
uwb_len = second * fps_uwb
bre_len = uwb_len
hrt_len = uwb_len
batch_size = 1
x_uwb = np.linspace(0, uwb_len-3, uwb_len-2)
x_bre = np.linspace(-1, 1, bre_len)
x_hrt = np.linspace(-1, 1, hrt_len)

fontsize = 25
filefolder = 'train'
model_path = 'ModelResult/All.pth'
text = "walking-person"

uwb_train_dir_A = os.path.join('FinalDataset', filefolder, 'UWB', 'A')
uwb_train_list_A = os.listdir(uwb_train_dir_A)
uwb_train_list_A = [tl for tl in uwb_train_list_A if text in tl]

uwb_train_dir_I = os.path.join('FinalDataset', filefolder, 'UWB', 'I')
uwb_train_list_I = os.listdir(uwb_train_dir_I)
uwb_train_list_I = [tl for tl in uwb_train_list_I if text in tl]

uwb_train_dir_Q = os.path.join('FinalDataset', filefolder, 'UWB', 'Q')
uwb_train_list_Q = os.listdir(uwb_train_dir_Q)
uwb_train_list_Q = [tl for tl in uwb_train_list_Q if text in tl]

resp_train_dir = os.path.join('FinalDataset', filefolder, 'Breath')
resp_train_list = os.listdir(resp_train_dir)
resp_train_list = [tl for tl in resp_train_list if text in tl]

# heart_train_dir = os.path.join('FinalDataset', filefolder, 'ECG')
# heart_train_list = os.listdir(heart_train_dir)
# heart_train_list = [tl for tl in heart_train_list if text in tl]

# Read the dataset
assert len(uwb_train_list_A) == len(uwb_train_list_I)
assert len(uwb_train_list_I) == len(uwb_train_list_Q)
assert len(uwb_train_list_I) == 2 * len(resp_train_list)
# assert len(uwb_train_list_Q) == 2 * len(heart_train_list)

sample_train = {}
for i in range(len(resp_train_list)):
    uwb_path_A_1 = os.path.join(uwb_train_dir_A, uwb_train_list_A[2 * i])
    tmp_A_1 = np.loadtxt(uwb_path_A_1, dtype=np.str_, delimiter=",")
    uwb_A_1 = tmp_A_1[:].astype(np.float32)
    uwb_A_1 = torch.from_numpy(uwb_A_1)
    uwb_A_1 = uwb_A_1.reshape(1, uwb_len)

    uwb_path_I_1 = os.path.join(uwb_train_dir_I, uwb_train_list_I[2 * i])
    tmp_I_1 = np.loadtxt(uwb_path_I_1, dtype=np.str_, delimiter=",")
    uwb_I_1 = tmp_I_1[:].astype(np.float32)
    uwb_I_1 = torch.from_numpy(uwb_I_1)
    uwb_I_1 = uwb_I_1.reshape(1, uwb_len)

    uwb_path_Q_1 = os.path.join(uwb_train_dir_Q, uwb_train_list_Q[2 * i])
    tmp_Q_1 = np.loadtxt(uwb_path_Q_1, dtype=np.str_, delimiter=",")
    uwb_Q_1 = tmp_Q_1[:].astype(np.float32)
    uwb_Q_1 = torch.from_numpy(uwb_Q_1)
    uwb_Q_1 = uwb_Q_1.reshape(1, uwb_len)
    uwb_1 = torch.cat((uwb_A_1, uwb_I_1), dim=0)
    uwb_1 = torch.cat((uwb_1, uwb_Q_1), dim=0)
    sample_train.update({'uwb_1' + str(i): uwb_1})

    uwb_path_A_2 = os.path.join(uwb_train_dir_A, uwb_train_list_A[2 * i + 1])
    tmp_A_2 = np.loadtxt(uwb_path_A_2, dtype=np.str_, delimiter=",")
    uwb_A_2 = tmp_A_2[:].astype(np.float32)
    uwb_A_2 = torch.from_numpy(uwb_A_2)
    uwb_A_2 = uwb_A_2.reshape(1, uwb_len)

    uwb_path_I_2 = os.path.join(uwb_train_dir_I, uwb_train_list_I[2 * i + 1])
    tmp_I_2 = np.loadtxt(uwb_path_I_2, dtype=np.str_, delimiter=",")
    uwb_I_2 = tmp_I_2[:].astype(np.float32)
    uwb_I_2 = torch.from_numpy(uwb_I_2)
    uwb_I_2 = uwb_I_2.reshape(1, uwb_len)

    uwb_path_Q_2 = os.path.join(uwb_train_dir_Q, uwb_train_list_Q[2 * i + 1])
    tmp_Q_2 = np.loadtxt(uwb_path_Q_2, dtype=np.str_, delimiter=",")
    uwb_Q_2 = tmp_Q_2[:].astype(np.float32)
    uwb_Q_2 = torch.from_numpy(uwb_Q_2)
    uwb_Q_2 = uwb_Q_2.reshape(1, uwb_len)
    # uwb_Q = torch.cat((uwb_Q_1, uwb_Q_2), dim=0)
    # sample_train.update({'uwb_Q' + str(i): uwb_Q})
    uwb_2 = torch.cat((uwb_A_2, uwb_I_2), dim=0)
    uwb_2 = torch.cat((uwb_2, uwb_Q_2), dim=0)
    sample_train.update({'uwb_2' + str(i): uwb_2})

    resp_item_path = os.path.join(resp_train_dir, resp_train_list[i])
    tmp = np.loadtxt(resp_item_path, dtype=np.str_, delimiter=",")
    resp = tmp[:].astype(np.float32)
    resp = torch.from_numpy(resp)
    resp = resp.reshape(1, bre_len)
    sample_train.update({'resp' + str(i): resp})

    # heart_item_path = os.path.join(heart_train_dir, heart_train_list[i])
    # tmp = np.loadtxt(heart_item_path, dtype=np.str_, delimiter=",")
    # heart = tmp[:].astype(np.float32)
    # heart = torch.from_numpy(heart)
    # heart = heart.reshape(1, hrt_len)
    # sample_train.update({'heart' + str(i): heart})

train_data_size = len(resp_train_list)
print('测试数据集的长度为：{}'.format(train_data_size))

# Model:
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

dataset = HealthDataset(sample_train, device, train_data_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# model = HealthDetector(uwb_len, bre_len, hrt_len).to(device)
# model = VAE(uwb_len, bre_len, hrt_len).to(device)

model = torch.load(model_path)

bre_sim_all = 0
hrt_sim_all = 0

bre_raw_all = 0
hrt_raw_all = 0

bre_raw_min = 100
bre_sim_min = 100

bre_raw_max = 0
bre_sim_max = 0

raw_error_num = 0
pred_error_num = 0

raw_error = 0
pred_error = 0

raw_error_rate = 0
pred_error_rate = 0

# Add tensorboard:
# writer = SummaryWriter("./logs_train")

def cos_similarity(a, b):
    d = torch.mul(a, b)
    a_len = torch.norm(a, dim=1)
    b_len = torch.norm(b, dim=1)
    cos = torch.sum(d, dim=1) / (a_len * b_len)


model.eval()

with torch.no_grad():
    # 测试步骤开始
    for uwb_1, uwb_2, bre_norm in dataloader:
        pred_bre, out_1, out_2 = model(uwb_1, uwb_2)
        pred_bre = torch.reshape(pred_bre, (1, -1))

        gt_norm = torch.reshape(bre_norm, (1, -1))
        gt_cut = gt_norm[:, 0:1478]
        bre_pred = torch.cosine_similarity(pred_bre, gt_cut)

        raw_cut = uwb_1[0, 0, 0:1478]
        raw_cut = torch.reshape(raw_cut, (1, -1))
        bre_raw = torch.cosine_similarity(raw_cut, gt_cut)

        bre_sim_all = bre_sim_all + bre_pred
        bre_raw_all = bre_raw_all + bre_raw


        # FFT
        gt_cut_fft = torch.fft.fft(gt_cut[0, 0:1478])
        gt_cut_abs = torch.abs(gt_cut_fft)
        raw_cut_fft = torch.fft.fft(uwb_1[0, 0, 0:1478])
        raw_cut_abs = torch.abs(raw_cut_fft)
        pre_cut_fft = torch.fft.fft(pred_bre[0, 0:1478])
        pre_cut_abs = torch.abs(pre_cut_fft)

        gt_rate = torch.argmax(gt_cut_abs[3:13])
        raw_rate = torch.argmax(raw_cut_abs[3:13])
        pre_cut_abs = torch.argmax(pre_cut_abs[3:13])

        raw_error = torch.abs(raw_rate-gt_rate)
        pred_error = torch.abs(pre_cut_abs - gt_rate)

        raw_error_num = raw_error_num + raw_error
        pred_error_num = pred_error_num + pred_error

        if bre_raw < bre_raw_min:
            bre_raw_min = bre_raw

        if bre_raw > bre_raw_max:
            bre_raw_max = bre_raw

        if bre_pred < bre_sim_min:
            bre_sim_min = bre_pred

        if bre_pred > bre_sim_max:
            bre_sim_max = bre_pred

        total_train_step = total_train_step + 1

        if total_train_step < 150:
            # print("{}".format(bre_raw.item()))
            print("{}".format(bre_pred.item()))

        # if total_train_step >= 0:
            # print(total_train_step)
            # pyplot.figure(dpi=100, figsize=(22, 7))
            # # pyplot.figure(dpi=100, figsize=(10, 5))
            # pyplot.plot(x_uwb, torch.detach(bre_norm[0, 0, 0:1478].to('cpu')).numpy(), '-', color='steelblue', label='GT data', linewidth=4)
            # pyplot.plot(x_uwb, torch.detach(uwb_1[0, 0, 0:1478].to('cpu')).numpy(), ':', color='grey', label='Processed data', linewidth=4)
            # pyplot.plot(x_uwb, torch.detach(pred_bre[0, 0:1478].to('cpu')).numpy(), '--', color='tomato', label='Prediction data', linewidth=4)
            # # pyplot.tick_params(labelsize=18)
            # font = {'size': fontsize}
            # pyplot.xlabel('Slow time', font)
            # pyplot.ylabel('Amplitude', font)
            # pyplot.xlim(0, 1477)
            # pyplot.ylim(0, 1)
            # pyplot.xticks(fontsize=fontsize)
            # pyplot.yticks(fontsize=fontsize)
            #
            # pyplot.rcParams.update({'font.size': fontsize})
            # pyplot.legend(bbox_to_anchor=(0.784095, 0.6))
            # pyplot.show()
        #
        #     pyplot.figure(num=2)
        #     pyplot.plot(x_uwb, torch.detach(gt_cut_abs).to('cpu').numpy())
        #     pyplot.plot(x_uwb, torch.detach(pre_cut_abs).to('cpu').numpy(), color='red')
        #     pyplot.plot(x_uwb, torch.detach(raw_cut_abs).to('cpu').numpy(), color='black')

        #     # writer.add_scalar("train_loss", loss.item(), total_train_step)

bre_sim_mean = bre_sim_all.item() / total_train_step
bre_raw_all = bre_raw_all.item() / total_train_step

raw_error_rate = raw_error_num / (total_train_step)
pred_error_rate = pred_error_num / (total_train_step)

print("Done")
print("Ours(Avg)：{}, Rate: {}".format(bre_sim_mean, pred_error_rate))
print("Raw(Avg)：{}, Rate: {}".format(bre_raw_all, raw_error_rate))
# print("Ours(平均)：{}, Ours(min): {}, Ours(max): {}, Rate: {}".format(bre_sim_mean, bre_sim_min, bre_sim_max, pred_error_rate))
# # print("Raw(平均)：{}, Raw(min): {}, Raw(max): {}, Rate: {}".format(bre_raw_all, bre_raw_min, bre_raw_max, raw_error_rate))
# tensorboard --logdir=logs_train --port=6006
# writer.close()
