import numpy as np
from numpy import abs, max, min, median
from scipy.fft import fft
from scipy.signal import correlate as xcorr

from process.mapminmax import mapminmax
from process.noiseEliminate import noiseEliminate


def dropdetect(CIRMatrix):
    print('dropdetect start')
    Inf = {'Filter': 'SubtractWindowMean'}
    CIRMatrix_NoiseReduced = noiseEliminate(Inf, CIRMatrix)
    CIRMatrix_Dropping = CIRMatrix_NoiseReduced
    win = 10
    d_win = win // 2
    f_win = 100
    thre = 1
    dropRecord = np.zeros_like(CIRMatrix_Dropping.T)
    std_bin = np.zeros(CIRMatrix_Dropping.shape[1])

    for i in range(CIRMatrix_Dropping.shape[1]):
        bins = CIRMatrix_Dropping[:, i]
        raw_sig = abs(bins)
        raw_sig = raw_sig.reshape(1, CIRMatrix_Dropping.shape[0])
        process_sig = np.zeros((1, raw_sig.shape[1]))
        dif = raw_sig[1:] - raw_sig[:-1]

        for j in range(d_win, len(raw_sig) - d_win):
            tmp = raw_sig[j - win // 2:j + win // 2 + 1]
            if max(tmp) == raw_sig[j]:
                dif_j = sum(abs(dif[j - d_win // 2:j + d_win // 2]))
                dif_round = sum(abs(dif[j - d_win:j + d_win]))
                dif_round -= dif_j
                dif_j /= (d_win // 2 * 2)
                dif_round /= (win - d_win // 2 * 2)
                if dif_j > thre * dif_round:
                    process_sig[j] = raw_sig[j]
        tmp = process_sig[process_sig != 0]
        if len(tmp) > 0:
            med = median(tmp)
            process_sig[process_sig < med] = 0
        process_sig = mapminmax(process_sig, 0, 1).reshape(1, CIRMatrix_Dropping.shape[0])
        dropRecord[i] = process_sig
        fft_all = fft(process_sig)
        fft_seg_1 = abs(fft_all[0, 19:135]).reshape(1, -1)
        fft_seg_2 = abs(fft_all[0, 135:251]).reshape(1, -1)
        corr_fft = xcorr(fft_seg_1, fft_seg_2, 'full', 'direct') / (
                len(fft_seg_1) - np.abs(np.arange(-len(fft_seg_1) + 1, len(fft_seg_1))))
        # 互相关无偏估计
        # # 计算全互相关
        # corr_full = xcorr(fft_seg_1[0], fft_seg_2[0], mode='full')
        # # 计算归一化因子进行无偏估计
        # # N = max(fft_seg_1.shape[1], fft_seg_2.shape[1])
        # N = 116
        # lags = np.arange(-N + 1, N)
        # normalization = N - np.abs(lags)
        # # 应用无偏估计的归一化
        # corr_fft = corr_full / normalization

        std_c = 0
        for u in range(1, len(corr_fft) - 1 - f_win):
            std_c += max(corr_fft[u:u + f_win]) - min(corr_fft[u:u + f_win])
        std_bin[i] = std_c

    dropIndex = np.argmax(std_bin[:5])
    drop_sig_p = dropRecord[dropIndex, :]
    drop_sig_r = CIRMatrix_Dropping[:, dropIndex]

    fft_rec_all = fft(drop_sig_p)
    record_fft = fft_rec_all[:1200]
    print('dropdetect end')
    return dropIndex, drop_sig_r, drop_sig_p
