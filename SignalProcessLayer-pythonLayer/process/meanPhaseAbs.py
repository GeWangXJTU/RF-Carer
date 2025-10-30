from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN

from process.scaleComplex import rotate_scale_complex__array, rotate_complex_numpy_array, scale_complex_array


def mean_angle_IQ_matrix(matrix):
    phase_matrix = np.angle(matrix)
    new_matrix = np.zeros_like(matrix)

    for col in range(phase_matrix.shape[1]):
        phase_now = phase_matrix[:, col]
        dbscan = DBSCAN(eps=0.005, min_samples=5)
        labels = dbscan.fit_predict(phase_now.reshape(-1, 1))
        label_counts = Counter(labels)
        max_label = max((label for label in label_counts if label != -1), key=label_counts.get)
        if max_label is not None and label_counts[max_label] > 0.3 * phase_matrix.shape[0]:
            cluster_values = phase_now[labels == max_label]
            cluster_mean = np.mean(cluster_values)
            IQ_now_adjust = rotate_complex_numpy_array(matrix[:, col], cluster_mean)
            new_matrix[:, col] = IQ_now_adjust
        else:
            IQ_now_adjust = rotate_complex_numpy_array(matrix[:, col], np.mean(phase_now))
            new_matrix[:, col] = IQ_now_adjust
    return new_matrix


def mean_angle_matrix(matrix):
    phase_matrix = np.angle(matrix)
    new_phase_matrix = np.zeros_like(phase_matrix)

    for col in range(phase_matrix.shape[1]):
        phase_now = phase_matrix[:, col]
        dbscan = DBSCAN(eps=0.005, min_samples=5)
        labels = dbscan.fit_predict(phase_now.reshape(-1, 1))
        label_counts = Counter(labels)
        max_label = max((label for label in label_counts if label != -1), key=label_counts.get)
        if max_label is not None and label_counts[max_label] > 0.3 * phase_matrix.shape[0]:
            cluster_values = phase_now[labels == max_label]
            cluster_mean = np.mean(cluster_values)
            phase_now_adjusted = phase_now - cluster_mean
            new_phase_matrix[:, col] = phase_now_adjusted
        else:
            phase_now_adjusted = phase_now - np.mean(phase_now)
            new_phase_matrix[:, col] = phase_now_adjusted
    return new_phase_matrix


def mean_abs_IQ_matrix(matrix):
    abs_matrix = np.abs(matrix)
    new_matrix = np.zeros_like(matrix)

    for col in range(abs_matrix.shape[1]):
        abs_now = abs_matrix[:, col]
        dbscan = DBSCAN(eps=0.005, min_samples=5)
        labels = dbscan.fit_predict(abs_now.reshape(-1, 1))
        label_counts = Counter(labels)
        max_label = max((label for label in label_counts if label != -1), key=label_counts.get)
        if max_label is not None and label_counts[max_label] > 0.3 * abs_matrix.shape[0]:
            cluster_values = abs_now[labels == max_label]
            cluster_mean = np.mean(cluster_values)
            IQ_now_adjust = scale_complex_array(matrix[:, col], cluster_mean)
            new_matrix[:, col] = IQ_now_adjust
        else:
            IQ_now_adjust = scale_complex_array(matrix[:, col], np.mean(abs_now))
            new_matrix[:, col] = IQ_now_adjust
    return new_matrix


def mean_abs_phase_IQ_matrix(matrix):
    abs_matrix = np.abs(matrix)
    phase_matrix = np.angle(matrix)
    new_matrix = np.zeros_like(matrix)

    for col in range(abs_matrix.shape[1]):
        abs_now = abs_matrix[:, col]
        dbscan1 = DBSCAN(eps=0.00001, min_samples=5)  # 0.00001
        labels1 = dbscan1.fit_predict(abs_now.reshape(-1, 1))
        label_counts1 = Counter(labels1)
        max_label1 = max((label for label in label_counts1 if label != -1), key=label_counts1.get)

        phase_now = phase_matrix[:, col]
        dbscan2 = DBSCAN(eps=0.005, min_samples=5)
        labels2 = dbscan2.fit_predict(phase_now.reshape(-1, 1))
        label_counts2 = Counter(labels2)
        max_label2 = max((label for label in label_counts2 if label != -1), key=label_counts2.get)

        cluster_values1 = abs_now[labels1 == max_label1]
        cluster_mean1 = np.mean(cluster_values1)

        cluster_values2 = phase_now[labels2 == max_label2]
        cluster_mean2 = np.mean(cluster_values2)

        if label_counts2[max_label2] > 0.3 * abs_matrix.shape[0] and label_counts1[max_label1] > 0.3 * abs_matrix.shape[
            0]:
            IQ_now_adjust = rotate_scale_complex__array(matrix[:, col], cluster_mean2, cluster_mean1)
        elif label_counts2[max_label2] > 0.3 * abs_matrix.shape[0] >= label_counts1[max_label1]:
            IQ_now_adjust = rotate_scale_complex__array(matrix[:, col], cluster_mean2, np.mean(abs_now))
        elif label_counts1[max_label1] > 0.3 * abs_matrix.shape[0] >= label_counts2[max_label2]:
            IQ_now_adjust = rotate_scale_complex__array(matrix[:, col], np.mean(phase_now), cluster_mean1)
        else:
            IQ_now_adjust = rotate_scale_complex__array(matrix[:, col], np.mean(phase_now), np.mean(abs_now))
        new_matrix[:, col] = IQ_now_adjust
    return new_matrix


def mean_IQ_abs_phase_matrix(matrix):
    abs_matrix = np.abs(matrix)
    phase_matrix = np.angle(matrix)
    use_matrix = np.copy(matrix)
    new_matrix = np.zeros_like(matrix)

    for col in range(abs_matrix.shape[1]):
        abs_now = abs_matrix[:, col]
        dbscan1 = DBSCAN(eps=0.00001, min_samples=5)  # 0.00001
        labels1 = dbscan1.fit_predict(abs_now.reshape(-1, 1))
        label_counts1 = Counter(labels1)
        max_label1 = max((label for label in label_counts1 if label != -1), key=label_counts1.get)

        phase_now = phase_matrix[:, col]
        dbscan2 = DBSCAN(eps=0.005, min_samples=5)
        labels2 = dbscan2.fit_predict(phase_now.reshape(-1, 1))
        label_counts2 = Counter(labels2)
        max_label2 = max((label for label in label_counts2 if label != -1), key=label_counts2.get)

        indices1 = np.where(labels1 == max_label1)[0]
        indices2 = np.where(labels2 == max_label2)[0]

        common_indices = np.intersect1d(indices1, indices2)

        if len(common_indices) > 0:
            mean_IQ = np.mean(use_matrix[common_indices, col])
            if common_indices.shape[0] > 0.3 * matrix.shape[0]:
                IQ_now_adjust = matrix[:, col] - mean_IQ
            else:
                IQ_now_adjust = matrix[:, col] - np.mean(matrix[:, col])
            new_matrix[:, col] = IQ_now_adjust
        else:
            IQ_now_adjust = matrix[:, col] - np.mean(matrix[:, col])
            new_matrix[:, col] = IQ_now_adjust
    return new_matrix


def mean_IQ_abs_phase_IQ_matrix(matrix):
    abs_matrix = np.abs(matrix)
    phase_matrix = np.angle(matrix)
    use_matrix = np.copy(matrix)
    new_matrix = np.zeros_like(matrix)

    for col in range(abs_matrix.shape[1]):
        abs_now = abs_matrix[:, col]
        dbscan1 = DBSCAN(eps=0.00001, min_samples=5)  # 0.00001
        labels1 = dbscan1.fit_predict(abs_now.reshape(-1, 1))
        label_counts1 = Counter(labels1)
        max_label1 = max((label for label in label_counts1 if label != -1), key=label_counts1.get)

        phase_now = phase_matrix[:, col]
        dbscan2 = DBSCAN(eps=0.005, min_samples=5)
        labels2 = dbscan2.fit_predict(phase_now.reshape(-1, 1))
        label_counts2 = Counter(labels2)
        max_label2 = max((label for label in label_counts2 if label != -1), key=label_counts2.get)

        cluster_values1 = abs_now[labels1 == max_label1]
        cluster_mean1 = np.mean(cluster_values1)

        cluster_values2 = phase_now[labels2 == max_label2]
        cluster_mean2 = np.mean(cluster_values2)

        indices1 = np.where(labels1 == max_label1)[0]
        indices2 = np.where(labels2 == max_label2)[0]

        common_indices = np.intersect1d(indices1, indices2)
        
        all_indices = np.arange(matrix.shape[0])
        complement_indices = np.setdiff1d(all_indices, common_indices)

        if len(common_indices) > 0:
            mean_IQ = np.mean(use_matrix[common_indices, col])
            if common_indices.shape[0] > 0.3 * matrix.shape[0]:
                
                new_matrix[complement_indices, col] = matrix[complement_indices, col] - mean_IQ
            else:
                new_matrix[complement_indices, col] = matrix[complement_indices, col] - np.mean(matrix[:, col])

            if label_counts2[max_label2] > 0.3 * abs_matrix.shape[0] and label_counts1[max_label1] > 0.3 * \
                    abs_matrix.shape[
                        0]:
                new_matrix[common_indices, col] = rotate_scale_complex__array(matrix[common_indices, col],
                                                                              cluster_mean2, cluster_mean1)
            elif label_counts2[max_label2] > 0.3 * abs_matrix.shape[0] >= label_counts1[max_label1]:
                new_matrix[common_indices, col] = rotate_scale_complex__array(matrix[common_indices, col],
                                                                              cluster_mean2, np.mean(abs_now))
            elif label_counts1[max_label1] > 0.3 * abs_matrix.shape[0] >= label_counts2[max_label2]:
                new_matrix[common_indices, col] = rotate_scale_complex__array(matrix[common_indices, col],
                                                                              np.mean(phase_now), cluster_mean1)
            else:
                new_matrix[common_indices, col] = rotate_scale_complex__array(matrix[common_indices, col],
                                                                              np.mean(phase_now), np.mean(abs_now))


        else:
            IQ_now_adjust = matrix[:, col] - np.mean(matrix[:, col])
            new_matrix[:, col] = IQ_now_adjust

    return new_matrix


def mean_IQ_abs_phase_IQ_matrix_2(matrix):
    abs_matrix = np.abs(matrix)
    phase_matrix = np.angle(matrix)
    use_matrix = np.copy(matrix)
    new_matrix = np.zeros_like(matrix)

    for col in range(abs_matrix.shape[1]):
        abs_now = abs_matrix[:, col]
        dbscan1 = DBSCAN(eps=0.00001, min_samples=5)  # 0.00001
        labels1 = dbscan1.fit_predict(abs_now.reshape(-1, 1))
        label_counts1 = Counter(labels1)
        max_label1 = max((label for label in label_counts1 if label != -1), key=label_counts1.get)

        phase_now = phase_matrix[:, col]
        dbscan2 = DBSCAN(eps=0.005, min_samples=5)
        labels2 = dbscan2.fit_predict(phase_now.reshape(-1, 1))
        label_counts2 = Counter(labels2)
        max_label2 = max((label for label in label_counts2 if label != -1), key=label_counts2.get)

        indices1 = np.where(labels1 == max_label1)[0]
        indices2 = np.where(labels2 == max_label2)[0]

        common_indices = np.intersect1d(indices1, indices2)
        
        all_indices = np.arange(matrix.shape[0])
        complement_indices = np.setdiff1d(all_indices, common_indices)

        if len(common_indices) > 0:
            mean_IQ = np.mean(use_matrix[common_indices, col])
            if common_indices.shape[0] > 0.3 * matrix.shape[0]:
                
                new_matrix[complement_indices, col] = matrix[complement_indices, col] - mean_IQ
            else:
                new_matrix[complement_indices, col] = matrix[complement_indices, col] - np.mean(matrix[:, col])

            new_matrix[common_indices, col] = 0 + 0j
        else:
            IQ_now_adjust = matrix[:, col] - np.mean(matrix[:, col])
            new_matrix[:, col] = IQ_now_adjust

    return new_matrix
