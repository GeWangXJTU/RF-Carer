import torch
import torch.nn as nn
import torch.nn.functional as F


class HealthLossFunc(nn.Module):
    def __init__(self):
        super(HealthLossFunc, self).__init__()
        self.loss1 = nn.MSELoss()
        self.loss2 = nn.KLDivLoss()
        # self.loss3 = nn.CrossEntropyLoss()
        return

    def forward(self, bre_out, out_1, out_2, resp, device='cuda', alpha=5):
        resp_c = resp[:, :, 0:-2]
        # heart_c = heart[:, :, 0:-2]
        resp_c = resp_c.reshape(1478)
        # heart_c = heart_c.reshape(1478)
        bre_out=bre_out.reshape(1478)
        # hrt_out=hrt_out.reshape(1478)
        loss_cons_resp = self.loss1(bre_out, resp_c)
        # loss_cons_heart = self.loss1(hrt_out, heart_c)

        # std_resp = torch.std(resp)
        # std_heart = torch.std(heart)
        # N = torch.normal(mean=0., std=std_resp+std_heart, size=[1478])
        # out_1 = out_1.reshape(-1)
        # out_2 = out_2.reshape(-1)
        # loss_regu_I = self.loss2(F.log_softmax(out_1, dim=-1), N)
        # loss_regu_Q = self.loss2(F.log_softmax(out_2, dim=-1), N)

        # loss = (loss_cons_resp + loss_cons_heart) + alpha * (
        #         loss_regu_I + loss_regu_Q)

        loss_regu = self.loss1(out_1, out_2)
        # loss = (0.05*loss_cons_heart+loss_cons_resp) + alpha * loss_regu
        #
        loss = loss_cons_resp+alpha * loss_regu
        return loss, loss_cons_resp, loss_regu
