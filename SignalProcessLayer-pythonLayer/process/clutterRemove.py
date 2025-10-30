import numpy as np


def EACR(matrix, lmd):
    R = np.abs(matrix)
    # for col in range(R.shape[1]):
    #     R[:, col] = (col + 1) * R[:, col]
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    C = np.zeros_like(R)

    C[0, :] = np.mean(R[0, 5:])
    for row in range(1, R.shape[0]):
        C[row, :] = lmd * C[row - 1, :] + (1 - lmd) * R[row, :]

    R_1 = R - C
    R_1 = np.maximum(R_1, 0)

    # R_1=np.abs(R - C)

    for col in range(R_1.shape[1]):
        R_1[:, col] = (col + 1) * R_1[:, col]

    R_1 = (R_1 - np.min(R_1)) / (np.max(R_1) - np.min(R_1))
    return R_1


def IACRA(matrix, lmdl, lmdh, xita):
    R = np.abs(matrix)
    # for col in range(R.shape[1]):
    #     R[:, col] = (col + 1) * R[:, col]
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    C = np.zeros_like(R)
    # xita = np.zeros_like(R)
    # xita[:, :] = 0.5
    # lmdL = 0.3
    # lmdH = 0.7
    C[0, :] = np.mean(R[0, 5:])

    for row in range(1, R.shape[0]):
        for col in range(R.shape[1]):
            prev_C = C[row - 1, col]
            prev_xita = xita[row - 1, col]
            diff = np.abs(R[row, col] - prev_C)

            if diff < 2 * prev_xita:
                C[row, col] = lmdl * prev_C + (1 - lmdl) * R[row, col]
                xita[row, col] = np.sqrt(
                    lmdl * prev_xita ** 2 + (1 - lmdl) * (R[row, col] - C[row, col]) ** 2)
            elif diff > 3 * prev_xita:
                C[row, col] = prev_C
                xita[row, col] = prev_xita
            else:
                C[row, col] = lmdh * prev_C + (1 - lmdh) * R[row, col]
                xita[row, col] = np.sqrt(
                    lmdh * prev_xita ** 2 + (1 - lmdh) * (R[row, col] - C[row, col]) ** 2)
    R_1 = R - C
    R_1 = np.maximum(R_1, 0)

    # R_1=np.abs(R - C)
    R_1 = (R_1 - np.min(R_1)) / (np.max(R_1) - np.min(R_1))
    for col in range(R_1.shape[1]):
        R_1[:, col] = (col + 1) * R_1[:, col]

    R_1 = (R_1 - np.min(R_1)) / (np.max(R_1) - np.min(R_1))
    return R_1
