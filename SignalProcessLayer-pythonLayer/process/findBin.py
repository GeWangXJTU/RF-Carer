from collections import defaultdict

import numpy as np
from scipy.signal import find_peaks

from process.meanPhaseAbs import mean_abs_phase_IQ_matrix
from process.pathSearch import dfs_search_window_3
from process.utils import dict_to_matrix, get_peak_indices_and_values


def find_bin_freq(CIRMatrix, sample, wid, windows, resp_a, resp_b, heart_a, heart_b):
    max_resp = np.zeros((1, wid - windows))
    max_heart = np.zeros((1, wid - windows))
    max_peo = np.ones((1, wid - windows))

    for i in range(wid - windows):
        realPlus = np.ones((sample, 1))
        imagPlus = np.ones((sample, 1))
        fft_peo = np.zeros((sample, 1))
        fft_mul = np.zeros((sample, 1))

        for j in range(windows):
            rawSig = CIRMatrix[:, i + j]

            realSig = np.real(rawSig)
            imagSig = np.imag(rawSig)

            realFFT = np.fft.fft(realSig).reshape(sample, 1)
            imagFFT = np.fft.fft(imagSig).reshape(sample, 1)

            realPlus *= np.abs(realFFT / sample)
            imagPlus *= np.abs(imagFFT / sample)
            fft_peo += realPlus * 2 + imagPlus * 2
            fft_mul += realPlus * 2 * imagPlus * 2

        max_resp[0, i] = np.max(fft_peo[resp_a:resp_b, 0])
        max_heart[0, i] = np.max(fft_peo[heart_a:heart_b, 0])
        max_peo[0, i] = np.max(fft_mul[heart_a:heart_b, 0])
    pos_rp = np.argsort(-max_resp)
    pos_ht = np.argsort(-max_heart)

    # Rpidx = pos_rp[(pos_rp >= 9) & (pos_rp <= 25)]
    Rpidx = pos_rp[(pos_rp >= 5) & (pos_rp <= 96)]
    htidx = pos_ht[(pos_ht >= 15) & (pos_ht <= 25)]

    Rp_idx = Rpidx[0]
    Ht_idx_1 = htidx[0]
    Ht_idx_2 = htidx[1]
    Ht_idx_3 = htidx[2]

    if max_resp[0, Rp_idx + 1] > max_resp[0, Rp_idx - 1]:
        Rp_idx_2 = Rp_idx + 1
    else:
        Rp_idx_2 = Rp_idx - 1

    RpIdxCIR_1 = Rp_idx + (windows // 2)
    RpIdxCIR_2 = Rp_idx_2 + (windows // 2)
    HtIdxCIR_1 = Ht_idx_1 + (windows // 2)
    HtIdxCIR_2 = Ht_idx_1 - 1
    HtIdxCIR_3 = Ht_idx_1 + 1
    return RpIdxCIR_1, RpIdxCIR_2


def find_bin_amp(CIRMatrix):
    peak_value_sum = defaultdict(float)

    for row in CIRMatrix:
        peaks, _ = find_peaks(np.abs(row))
        for peak_index in peaks:
            peak_value_sum[peak_index] += np.abs(row[peak_index])

    sorted_peaks = sorted(peak_value_sum.keys(), key=lambda x: peak_value_sum[x], reverse=True)
    filtered_peaks = [peak for peak in sorted_peaks if peak >= 5]
    return sorted_peaks, filtered_peaks


def find_bin_path(CIRMatrix):
    magnitude = np.abs(mean_abs_phase_IQ_matrix(CIRMatrix))
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    for col in range(magnitude.shape[1]):
        magnitude[:, col] = (col + 1) * magnitude[:, col]
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    magnitude[magnitude < 0.05] = 0
    peaks_per_row = get_peak_indices_and_values(magnitude)
    indices_set = list(set(np.concatenate([cols['indices'] for cols in peaks_per_row.values()])))
    mask_peaks = dict_to_matrix(peaks_per_row, shape=(2000, 96))
    mask_peaks = (mask_peaks - np.min(mask_peaks)) / (np.max(mask_peaks) - np.min(mask_peaks))
    mask_peaks_1 = np.copy(mask_peaks)
    mask_peaks_2 = np.copy(mask_peaks)
    result_path = dfs_search_window_3(mask_peaks, 0.1, 0.5)
    result_path_2 = dfs_search_window_3(mask_peaks_1, 0.075, 0.5)
    result_path_3 = dfs_search_window_3(mask_peaks_2, 0.05, 0.5)
    return result_path, result_path_2, result_path_3


def find_bin_threshold(CIRMatrix):
    magnitude = np.abs(mean_abs_phase_IQ_matrix(CIRMatrix))

    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

    for col in range(magnitude.shape[1]):
        magnitude[:, col] = (col + 1) * magnitude[:, col]

    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

    magnitude[magnitude < 0.05] = 0

    peaks_per_row = get_peak_indices_and_values(magnitude)
    indices_set = list(set(np.concatenate([cols['indices'] for cols in peaks_per_row.values()])))

    mask_peaks = dict_to_matrix(peaks_per_row, shape=(2000, 96))
    mask_peaks = (mask_peaks - np.min(mask_peaks)) / (np.max(mask_peaks) - np.min(mask_peaks))

    A = np.copy(mask_peaks)

    C = np.zeros((1999, 96, 96))

    alpha = 15  # 差值权重
    beta = 5  # 位置差权重
    gamma = 5  # 和值权重

    for i in range(1999):
        # 获取当前行和下一行
        A_i = A[i, :].reshape(96, 1)  # 当前行 i 转换为列向量
        A_i_plus_1 = A[i + 1, :].reshape(1, 96)  # 下一行 i+1 转换为行向量
        term1 = np.abs(A_i - A_i_plus_1)
        l_indices = np.arange(96).reshape(96, 1)  # 列索引 l
        m_indices = np.arange(96).reshape(1, 96)  # 行索引 m
        # term2 = np.log10(np.abs(m_indices - l_indices) + 1)
        term2 = np.abs(m_indices - l_indices)
        term3 = 1 / np.abs(A_i + A_i_plus_1 + 1e-16)
        C[i, :, :] = alpha * term1 + beta * term2 + gamma * term3

    np.set_printoptions(suppress=True)

    start = np.argmax(mask_peaks[0, :])
    indices_list = [start]
    for i in range(1999):
        candidate_indice = np.argmin(C[i, start, :])
        candidate = np.min(C[i, start, :])
        if np.all(C[i, start, :] == candidate):
            indices_list.append(start)
            start = start
        else:
            indices_list.append(candidate_indice)
            start = candidate_indice

    indices_result = list(enumerate(indices_list))
    indices_result_path_1 = {"path_1": indices_result}

    return indices_result
