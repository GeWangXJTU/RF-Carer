import numpy as np


def eliminateError(sig, win):
    sig = np.array(sig)
    length = len(sig)
    sig_pro = np.zeros(length)

    for i in range(length):
        if i <= win // 2:
            me = np.mean(sig[:i + win // 2 + 1])
        elif i > length - win // 2 - 1:
            me = np.mean(sig[i - win // 2:])
        else:
            me = np.mean(sig[i - win // 2:i + win // 2 + 1])

        sig_pro[i] = sig[i] - me

    return sig_pro

