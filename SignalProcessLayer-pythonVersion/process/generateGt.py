import numpy as np
from scipy import signal

from process.extendAndShift import extendAndShift
from process.removeError import removeError


def generate_gt(CIRMatrix, BreathData, TimeStamps):
    print('generate_gt start')
    str_second = 0
    en_second = 50

    seconds = 50
    fps_uwb = 40
    fps_bre = 20
    sample = fps_uwb * seconds
    sample_bre = fps_bre * seconds

    Breath_sig = []
    for i in range(len(BreathData)):
        if type(BreathData[i]) != list:
            Breath_sig.append(BreathData[i])
        else:
            Breath_sig.extend(BreathData[i])
    Breath_sig = list(map(float, Breath_sig))

    Breath_sig = [x for x in Breath_sig if x != -1000]
    Breath_sig_pro = removeError(Breath_sig)
    flag = 1

    if len(Breath_sig_pro) == sample_bre:
        flag += 1
    elif len(Breath_sig_pro) > sample_bre:
        Breath_sig_pro = Breath_sig_pro[:sample_bre]
        flag += 1
    elif sample_bre - 30 < len(Breath_sig_pro) < sample_bre:
        Breath_sig_pro = np.pad(Breath_sig_pro, (0, sample_bre - len(Breath_sig_pro)), 'constant')
        flag += 1
    else:
        Breath_sig_pro = extendAndShift(Breath_sig_pro, sample_bre)
        flag += 1
    if flag == 2:
        Breath_num_samples = int(len(Breath_sig_pro) * sample / sample_bre)
        Breath_sig_final = signal.resample(Breath_sig_pro, Breath_num_samples)
        Breath_sig_final = Breath_sig_final[str_second * fps_uwb:en_second * fps_uwb]
    else:
        Breath_sig_final = 0
    print('generate_gt end')
    return Breath_sig_final


