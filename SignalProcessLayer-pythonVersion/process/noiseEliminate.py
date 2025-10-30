import numpy as np


def noiseEliminate(Inf, CIRMatrix):
    print('noiseEliminate start')
    if Inf['Filter'] == 'SubtractSelfMean':
        CIRMatrix_NoiseReduced = CIRMatrix - np.mean(CIRMatrix, axis=0, keepdims=True)

    elif Inf['Filter'] == 'SubtractLastFrame':
        CIRMatrix_NoiseReduced = CIRMatrix - np.vstack([CIRMatrix[0, :], CIRMatrix[:-1, :]])

    elif Inf['Filter'] == 'SubtractWindowMean':
        win = 10
        num_frames, num_columns = CIRMatrix.shape
        ReferenceMatrix = np.zeros_like(CIRMatrix)
        for i in range(num_frames):
            window_start = max(0, i - win // 2)
            window_end = min(num_frames, i + win // 2 + 1)
            Window = CIRMatrix[window_start:window_end, :]
            ReferenceFrame = np.mean(Window, axis=0)
            ReferenceMatrix[i, :] = ReferenceFrame
        CIRMatrix_NoiseReduced = CIRMatrix - ReferenceMatrix
        print('noiseEliminate end')
    return CIRMatrix_NoiseReduced
