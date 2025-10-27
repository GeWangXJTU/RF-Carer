import torch
import math

def calRate(sig, win=200, step=50, device='cuda', thre = 0.5):
    idx = sig.size()[0]
    length = math.floor(idx/step)
    max_index = torch.Tensor(length).to(device)
    for i in range(length):
        str = i * step
        en = str + win
        cut_sig = sig[str:en]
        # max_index = sg.argrelmax(cut_sig.cpu().numpy())[0]
        tmp = torch.argmax(cut_sig) + str
        val = torch.max(cut_sig).item()
        if (tmp > str and tmp < en and val > thre):
            max_index[i] = tmp.to(device) + torch.tensor([str]).to(device)

    return max_index
