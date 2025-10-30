import os

import numpy as np

from process.processGT import processGT


os.environ["OMP_NUM_THREADS"] = "5"
np.set_printoptions(threshold=np.inf)


def preProcessVital(start, end, cut_sample, CIRMatrix, Breath_sig_final):
    print('preProcessVital start')

    sample, wid = CIRMatrix.shape
    windows = 3
    Fs = 40
    heartRate = [1, 2]
    respRate = [0.2, 0.4]  # 对应[2.4, 18]
    f = Fs * np.arange(sample // 2 + 1) / sample

    heart_a = np.min(np.where(f >= heartRate[0]))
    heart_b = np.max(np.where(f <= heartRate[1]))
    resp_a = np.min(np.where(f >= respRate[0]))
    resp_b = np.max(np.where(f <= respRate[1]))

    gap = 400
    f_gap = Fs * np.arange(gap // 2 + 1) / gap
    gap_a = np.min(np.where(f_gap >= heartRate[0]))
    gap_b = np.max(np.where(f_gap <= heartRate[1]))

    breath_gt = processGT(Breath_sig_final, heart_a, heart_b, gap, gap_a, gap_b)

    ####
    # signalCompensate
    ####

    ####
    # clutterMove
    ####

    ####
    # meanPhaseAbs
    ####

    ####
    # findBin
    ####

    ####
    # transformVector
    ####

    ####
    # dualRotate
    ####
