import torch.utils.data as data
import torch
import numpy as np
from matplotlib import pyplot


class HealthDataset(data.Dataset):
    def __init__(self, sample_train, device, train_data_size):
        self.sample_train = sample_train
        self.device = device
        self.len = train_data_size
        self.x_uwb = np.linspace(-1, 1, 37 * 40)
        self.x_bre = np.linspace(-1, 1, 37*50)
        self.x_hrt = np.linspace(-1, 1, 37*125)

    def __getitem__(self, index):
        uwb_1 = self.sample_train['uwb_1' + str(index)]
        uwb_2 = self.sample_train['uwb_2' + str(index)]
        uwb_1, uwb_2 = uwb_1.to(self.device), uwb_2.to(self.device)

        resp = self.sample_train['resp' + str(index)]
        # heart = self.sample_train['heart' + str(index)]
        # resp, heart = resp.to(self.device), heart.to(self.device)
        resp = resp.to(self.device)

        return uwb_1, uwb_2, resp

    def __len__(self):
        return self.len


