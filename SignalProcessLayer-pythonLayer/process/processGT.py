import numpy as np

from process.mapminmax import mapminmax


def processGT(Breath, heart_a, heart_b, gap, gap_a, gap_b):
    len_breath = len(Breath)
    len_breath = len_breath / 1
    breath_gt = np.zeros((1, int(len_breath)))
    for i in range(1):
        tmp = Breath[int(i * len_breath):int((i + 1) * len_breath)]
        tmp = mapminmax(tmp, 0, 1)
        breath_gt[i, :] = tmp

    return breath_gt
