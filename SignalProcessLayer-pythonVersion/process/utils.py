from collections import Counter

import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, butter, lfilter
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from statsmodels.tsa.stattools import acf


def dict_to_matrix(data_dict, shape=None):
    if not data_dict:
        return np.array([[]])

    if shape is None:
        max_row = max(data_dict.keys())
        max_col = max(max(cols['indices']) for cols in data_dict.values()) if data_dict else 0

        matrix = np.zeros((max_row + 1, max_col + 1), dtype=float)
    else:
        max_row, max_col = shape
        matrix = np.zeros((max_row, max_col), dtype=float)

    for row, cols_dict in data_dict.items():
        indices = cols_dict['indices']
        values = cols_dict['values']

        for idx, col in enumerate(indices):
            matrix[row, col] = values[idx]

    return matrix


def get_peak_indices_and_values(matrix):
    peaks_per_row = {}

    for row_index in range(matrix.shape[0]):
        peaks, _ = find_peaks(matrix[row_index, :])
        peaks_per_row[row_index] = {
            "indices": peaks,
            "values": matrix[row_index, peaks]
        }

    return peaks_per_row


def corr_group(resultsIQP, correlation_threshold):
    path_ids = list(resultsIQP.keys())

    correlation_pairs = []

    for i in range(len(path_ids)):
        for j in range(i + 1, len(path_ids)):
            path_id_1 = path_ids[i]
            path_id_2 = path_ids[j]

            P1_1 = resultsIQP[path_id_1]['record_P1'][0, :]
            P1_2 = resultsIQP[path_id_2]['record_P1'][0, :]

            correlation = np.corrcoef(P1_1, P1_2)[0, 1]

            correlation_pairs.append((path_id_1, path_id_2, correlation))

    group_sets_dict = {f'set{i + 1}': set() for i in range(len(path_ids))}
    for i, s in enumerate(group_sets_dict.keys()):
        group_sets_dict[s].add(path_ids[i])
    for path1, path2, correlation in correlation_pairs:

        if np.abs(correlation) > correlation_threshold:
            set_index1 = path_ids.index(path1)
            set_index2 = path_ids.index(path2)

            group_sets_dict[f'set{set_index1 + 1}'].add(path2)
            group_sets_dict[f'set{set_index2 + 1}'].add(path1)

    keys_to_remove = set()

    keys = list(group_sets_dict.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]

            if group_sets_dict[key1] == group_sets_dict[key2]:
                keys_to_remove.add(key2)

    for key in keys_to_remove:
        del group_sets_dict[key]

    return group_sets_dict


def bandpass_abs(extracted_data):
    fs = 40.0
    t = np.arange(0, 50, 1 / fs)

    order = 2  # 2,3,4
    nyquist = 0.5 * fs
    low = 0.2 / nyquist  # 0.08
    high = 0.4 / nyquist  # 0.4,0.6
    b, a = butter(order, [low, high], btype='bandpass', analog=False)

    filtered_signal = lfilter(b, a, extracted_data)

    return filtered_signal


def bandpass_phase(extracted_phase):
    fs = 40.0
    t = np.arange(0, 50, 1 / fs)

    order = 2  # 2,3,4
    nyquist = 0.5 * fs
    low = 0.2 / nyquist
    high = 0.4 / nyquist
    b, a = butter(order, [low, high], btype='bandpass', analog=False)

    filtered_data = {}

    for path_id, signal in extracted_phase.items():
        filtered_signal = lfilter(b, a, signal)
        filtered_data[path_id] = filtered_signal

    return filtered_data


def unwrap_angle(extracted_data):
    result_dict = {}

    for path_id, datas in extracted_data.items():
        y = np.unwrap(np.angle(datas))

        dbscan = DBSCAN(eps=0.02, min_samples=5)
        labels = dbscan.fit_predict(y.reshape(-1, 1))

        label_counts = Counter(labels)

        max_cluster_label = max((label for label in label_counts if label != -1), key=lambda label: label_counts[label])
        noise_count = label_counts.get(-1, 0)

        largest_cluster_mask = labels == max_cluster_label
        largest_cluster_data = y[largest_cluster_mask]

        if noise_count + len(largest_cluster_data) == len(y):
            x_cluster = np.arange(len(largest_cluster_data))
            y_cluster = largest_cluster_data
            spline = UnivariateSpline(x_cluster, y_cluster)
            spline.set_smoothing_factor(5)
            x_new = np.linspace(x_cluster.min(), x_cluster.max(), len(largest_cluster_data))
            y_spline = spline(x_new)
            if np.abs(np.corrcoef(y_spline, y_cluster)[0, 1]) <= 0.5:
                spline.set_smoothing_factor(0.3)
                y_spline = spline(x_new)
            largest_cluster_data_num_samples = len(y)
            unwrap_angle_data = signal.resample(y_spline, largest_cluster_data_num_samples)
        else:
            largest_cluster_data_num_samples = len(y)
            unwrap_angle_data = signal.resample(largest_cluster_data, largest_cluster_data_num_samples)

        result_dict[path_id] = unwrap_angle_data

    return result_dict


def apply_window_average(matrix, window_size1, window_size2):
    rows, cols = matrix.shape

    offset1 = window_size1 // 2
    offset2 = window_size2 // 2
    result = np.zeros_like(matrix, dtype=float)

    for i in range(rows):
        for j in range(cols):
            row_start = max(i - offset1, 0)
            row_end = min(i + offset1 + 1, rows)
            col_start = max(j - offset2, 0)
            col_end = min(j + offset2 + 1, cols)

            window = matrix[row_start:row_end, col_start:col_end]

            result[i, j] = np.mean(window)

    return result


def cos_sim(original, fitted):
    numerator = np.dot(original, fitted)
    denominator = np.linalg.norm(original) * np.linalg.norm(fitted)
    cos_similarity = numerator / denominator
    return cos_similarity


def sinusoidal(x, a, b, c, d):
    return a * np.sin(b * x + c) + d


def check_col_circle(mask, data):
    result_data = np.copy(data)
    result_mask = np.copy(mask)

    sum_mask = np.sum(mask, axis=0)
    mask_miji_indices = np.where(sum_mask >= 10 * (data.shape[0] / 2))[0]
    mask_xishu_indices = np.where(sum_mask < 10 * (data.shape[0] / 2))[0]
    mask_zero_indices = np.where(sum_mask == 0)[0]
    mask_xishu_indices = np.setdiff1d(mask_xishu_indices, mask_zero_indices)

    lags = data.shape[0] - 1

    for col_idx in mask_miji_indices:
        autocorr = acf(zscore(np.unwrap(np.angle(data[:, col_idx]))), nlags=lags)

        peaks, _ = find_peaks(autocorr, distance=40)

        valid_peaks = [peak for peak in peaks if autocorr[peak] > 0][:4]
        if valid_peaks:
            intervals = np.diff(valid_peaks)
            if len(valid_peaks) >= 4:
                if np.all(np.abs(intervals - intervals[0]) <= 0.175 * intervals[0]):
                    result_mask[:, col_idx] = np.where(result_mask[:, col_idx] == 10, 9, result_mask[:, col_idx])
                elif np.all(np.abs(intervals - intervals[0]) > 0.175 * intervals[0]) and np.all(
                        np.abs(intervals - intervals[0]) <= 0.25 * intervals[0]):
                    result_mask[:, col_idx] = np.where(result_mask[:, col_idx] == 10, 6, result_mask[:, col_idx])
                else:
                    result_mask[:, col_idx] = np.where(result_mask[:, col_idx] == 10, 3, result_mask[:, col_idx])
            else:
                result_mask[:, col_idx] = np.where(result_mask[:, col_idx] == 10, 1, result_mask[:, col_idx])
        else:
            result_mask[:, col_idx] = np.where(result_mask[:, col_idx] == 10, 1, result_mask[:, col_idx])

    return result_mask, result_data, mask_miji_indices, mask_xishu_indices


def check_col(mask, data):
    result_mask = np.copy(mask)

    non_zero_counts = np.count_nonzero(mask, axis=0)

    columns_above_threshold = np.where(non_zero_counts > (data.shape[0] * 0.6))[0]

    lags = data.shape[0] - 1

    for col_idx in columns_above_threshold:
        autocorr = acf(zscore(np.unwrap(np.angle(data[:, col_idx]))), nlags=lags)

        peaks, _ = find_peaks(autocorr, distance=40)

        valid_peaks = [peak for peak in peaks if autocorr[peak] > 0][:4]
        if valid_peaks:
            intervals = np.diff(valid_peaks)
            if len(valid_peaks) >= 4:
                if np.all(np.abs(intervals - intervals[0]) <= 0.175 * intervals[0]):
                    result_mask[:, col_idx] = np.where(result_mask[:, col_idx] != 0, result_mask[:, col_idx] * 1,
                                                       result_mask[:, col_idx])
                elif np.all(np.abs(intervals - intervals[0]) > 0.175 * intervals[0]) and np.all(
                        np.abs(intervals - intervals[0]) <= 0.25 * intervals[0]):
                    result_mask[:, col_idx] = np.where(result_mask[:, col_idx] != 0, result_mask[:, col_idx] * 0.9,
                                                       result_mask[:, col_idx])
                else:
                    result_mask[:, col_idx] = np.where(result_mask[:, col_idx] != 0, result_mask[:, col_idx] * 0.1,
                                                       result_mask[:, col_idx])
            else:
                result_mask[:, col_idx] = np.where(result_mask[:, col_idx] != 0, result_mask[:, col_idx] * 0.1,
                                                   result_mask[:, col_idx])
        else:
            result_mask[:, col_idx] = np.where(result_mask[:, col_idx] != 0, result_mask[:, col_idx] * 0.1,
                                               result_mask[:, col_idx])

    return result_mask


def are_zeroes_complementary(arr1, arr2):
    if len(arr1) != len(arr2):
        return False

    for a, b in zip(arr1, arr2):
        if (a == 0 and b == 0) or (a != 0 and b != 0):
            return False
    return True


def are_zeroes_complementary_three(arr1, arr2, arr3):
    if len(arr1) != len(arr2) or len(arr1) != len(arr3):
        return False

    for a, b, c in zip(arr1, arr2, arr3):
        non_zero_count = (a != 0) + (b != 0) + (c != 0)
        if non_zero_count != 1:
            return False
    return True


def check_hubu(mask, data, indices):
    result_data = np.copy(data)
    result_mask = np.copy(mask)
    indices = np.array(indices)
    for col_idx in indices:
        if col_idx + 1 in indices:
            if are_zeroes_complementary(mask[:, col_idx], mask[:, col_idx + 1]):
                if np.count_nonzero(mask[:, col_idx]) >= data.shape[0] * 0.9:
                    result_mask[:, col_idx] = [b if b != 0 else a for a, b in
                                               zip(result_mask[:, col_idx + 1], result_mask[:, col_idx])]
                    result_mask[:, col_idx + 1] = 0
                    indices = indices[indices != col_idx + 1]
                elif np.count_nonzero(mask[:, col_idx + 1]) >= data.shape[0] * 0.9:
                    result_mask[:, col_idx + 1] = [b if b != 0 else a for a, b in
                                                   zip(result_mask[:, col_idx], result_mask[:, col_idx + 1])]
                    result_mask[:, col_idx] = 0
                    indices = indices[indices != col_idx]
                else:
                    non_zero_value = np.sum(mask[:, col_idx]) / np.count_nonzero(mask[:, col_idx])
                    result_mask[:, col_idx][result_mask[:, col_idx] == 0] = non_zero_value
                    non_zero_value = np.sum(mask[:, col_idx + 1]) / np.count_nonzero(mask[:, col_idx + 1])
                    result_mask[:, col_idx + 1][result_mask[:, col_idx + 1] == 0] = non_zero_value

    return result_mask, result_data


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
