import os
import random

import numpy as np
import torch.optim
from matplotlib import pyplot
from torch import nn
from torch.utils.data import DataLoader

from Dataset import HealthDataset

# from torch.utils.tensorboard import SummaryWriter
# from model import HealthDetector
from ResModel import HealthDetector
from HealthLossFunc import HealthLossFunc

# Parameters:
total_train_step = 0
epoch = 100  # Number of round
learning_rate = 0.001
second = 37
fps_uwb = 40
fps_ecg = 125
fps_bre = 50
uwb_len = second * fps_uwb
bre_len = uwb_len
hrt_len = uwb_len
batch_size = 1
x_uwb = np.linspace(-1, 1, uwb_len - 2)
x_bre = np.linspace(-1, 1, bre_len)
x_hrt = np.linspace(-1, 1, hrt_len)

# model_path = 'ModelResult/test/ResModel_decay1e-2_50.pth'
save_path = 'ModelResult/All.pth'
text = "_agl"

uwb_train_dir_A = os.path.join('FinalDataset', 'train', 'UWB', 'A')
uwb_train_list_A = os.listdir(uwb_train_dir_A)
uwb_train_list_A = [tl for tl in uwb_train_list_A if text in tl]

uwb_train_dir_I = os.path.join('FinalDataset', 'train', 'UWB', 'I')
uwb_train_list_I = os.listdir(uwb_train_dir_I)
uwb_train_list_I = [tl for tl in uwb_train_list_I if text in tl]

uwb_train_dir_Q = os.path.join('FinalDataset', 'train', 'UWB', 'Q')
uwb_train_list_Q = os.listdir(uwb_train_dir_Q)
uwb_train_list_Q = [tl for tl in uwb_train_list_Q if text in tl]

resp_train_dir = os.path.join('FinalDataset', 'train', 'Breath')
resp_train_list = os.listdir(resp_train_dir)
resp_train_list = [tl for tl in resp_train_list if text in tl]

# heart_train_dir = os.path.join('FinalDataset', 'train', 'ECG')
# heart_train_list = os.listdir(heart_train_dir)
# heart_train_list = [tl for tl in heart_train_list if text not in tl]

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
print('The length of training data is：{}'.format(train_data_size))

# Model:
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
# x_uwb = torch.tensor(x_uwb).to(device)

dataset = HealthDataset(sample_train, device, train_data_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
model = HealthDetector(uwb_len, bre_len, hrt_len, device).to(device)
# model = torch.load(model_path)

# Loss Function:
LossFunction = HealthLossFunc().to(device)
# LossFunction = VARLossFunc().to(device)

# optimizer:
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01 * learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Add tensorboard:
# writer = SummaryWriter("./logs_train")

# initial weight
def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    elif type(layer) == nn.Linear:
        nn.init.constant_(layer.weight, 0.01)


# model.apply(init_weights)

for i in range(epoch):
    print("-------The {} -th training-------".format(i + 1))

    # 训练步骤开始

    model.train()
    for uwb_1, uwb_2, bre_norm in dataloader:

        pred_bre, out_1, out_2 = model(uwb_1, uwb_2)
        loss, loss_cons_resp, loss_com = LossFunction(pred_bre, out_1, out_2, bre_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
        if total_train_step % 200 == 0:
            # pyplot.figure(num=1)
            # pyplot.plot(x_uwb, torch.detach(hrt_norm[0, 0, 0:1478].to('cpu')).numpy())
            # pyplot.plot(x_uwb, torch.detach(pred_Hrt[0:1478].to('cpu')).numpy(), color='red')
            # pyplot.figure(num=2)
            # pyplot.plot(x_uwb, torch.detach(bre_norm[0, 0, 0:1478].to('cpu')).numpy())
            # pyplot.plot(x_uwb, torch.detach(pred_bre[0, 0, 0:1478].to('cpu')).numpy(), color='red')
            # pyplot.plot(x_uwb, torch.detach(uwb_1[0, 0, 0:1478].to('cpu')).numpy(), color='black')
            # pyplot.show()
            print("Training：{}, Loss: {}".format(total_train_step, loss.item()))
            print("Breath：{}, comp: {}".format(loss_cons_resp.item(), loss_com.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

torch.save(model, save_path)
print("Model saved")

# tensorboard --logdir=logs_train --port=6006
# writer.close()
