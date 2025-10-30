import numpy as np


def removeError(sig):
    print('removeError start')
    sig = np.array(sig)
    length = len(sig)
    thre = 256 * 0.8
    pre = sig[0]
    nxt = sig[2]
    for i in range(length):
        sig_high = sig[i] // 256
        sig_low = sig[i] % 256

        if sig_low == 170:
            if abs(pre - sig[i]) > thre:
                pre_high = pre // 256
                pre_low = pre % 256
                tmp = pre_high * 256 + sig_high

                if abs(tmp - pre) < thre:
                    sig[i] = tmp
                else:
                    if tmp > pre:
                        sig[i] = tmp - 256
                    else:
                        sig[i] = tmp + 256
            elif abs(nxt - sig[i]) > thre:
                pass
    print('removeError end')
    return sig
