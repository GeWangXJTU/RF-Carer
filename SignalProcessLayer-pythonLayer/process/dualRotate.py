import numpy as np
from process.mapminmax import mapminmax


def DualRotate(sig1, sig2, raw_sig1, raw_sig2, sample, breath_gt, raw_phase_1, raw_phase_2):
    print('DualRotate start')
    wid, len_ = sig1.shape
    record_I1 = np.zeros((2 * wid, len_))
    record_Q1 = np.zeros((2 * wid, len_))
    record_P1 = np.zeros((2 * wid, len_))
    record_I2 = np.zeros((2 * wid, len_))
    record_Q2 = np.zeros((2 * wid, len_))
    record_P2 = np.zeros((2 * wid, len_))

    str_ = -1 * np.pi

    window = 10
    len_sig = len(raw_sig1)
    agl_sig1 = np.angle(raw_sig1)
    agl_sig2 = np.angle(raw_sig2)
    agl_sig1 = np.unwrap(agl_sig1)
    agl_sig2 = np.unwrap(agl_sig2)

    pro_sig1 = np.copy(agl_sig1)
    pro_sig2 = np.copy(agl_sig2)

    for x in range(window // 2, len_sig - window // 2):
        pro_sig1[x] = np.mean(agl_sig1[x - window // 2:x + window // 2 + 1])
        pro_sig2[x] = np.mean(agl_sig2[x - window // 2:x + window // 2 + 1])

    for u in range(wid):
        cut1 = sig1[u, :]
        cut2 = sig2[u, :]
        ref1 = raw_phase_1
        ref2 = raw_phase_2
        ref = breath_gt[u, :]
        I_sig1 = np.real(cut1)
        Q_sig1 = np.imag(cut1)
        I_sig2 = np.real(cut2)
        Q_sig2 = np.imag(cut2)

        num_points = 720
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        rotate = []

        # 遍历每个可能的旋转角度
        for angle in angles:
            # 对信号1进行旋转
            tmp_I1 = I_sig1 * np.cos(angle) + Q_sig1 * np.sin(angle)
            tmp_Q1 = -I_sig1 * np.sin(angle) + Q_sig1 * np.cos(angle)

            # 对信号2进行旋转
            tmp_I2 = I_sig2 * np.cos(angle) + Q_sig2 * np.sin(angle)
            tmp_Q2 = -I_sig2 * np.sin(angle) + Q_sig2 * np.cos(angle)

            # 计算信号1的幅度范围平方和
            range_I1 = np.max(tmp_I1) - np.min(tmp_I1)
            range_Q1 = np.max(tmp_Q1) - np.min(tmp_Q1)
            value1 = range_I1 ** 2 + range_Q1 ** 2

            # 计算信号2的幅度范围平方和
            range_I2 = np.max(tmp_I2) - np.min(tmp_I2)
            range_Q2 = np.max(tmp_Q2) - np.min(tmp_Q2)
            value2 = range_I2 ** 2 + range_Q2 ** 2

            # 将两个信号的目标函数值相加
            rotate.append(value1 + value2)

        idx = np.argmax(rotate)
        theta_prime = angles[idx]
        I = I_sig1 * np.cos(theta_prime) + Q_sig1 * np.sin(theta_prime)
        Q = -I_sig1 * np.sin(theta_prime) + Q_sig1 * np.cos(theta_prime)
        min_val, min_idx = min(ref[:]), np.argmin(ref[:])
        max_val, max_idx = max(ref[:]), np.argmax(ref[:])

        corcof_I = I[max_idx] - I[min_idx]
        corcof_Q = Q[max_idx] - Q[min_idx]

        flag_I = -1 if corcof_I < 0 else 1
        flag_Q = -1 if corcof_Q < 0 else 1

        phase1 = np.unwrap(np.angle(flag_I * I + 1j * flag_Q * Q))
        phase1_1 = np.copy(phase1)
        for x in range(window // 2, len(phase1) - window // 2):
            phase1_1[x] = np.mean(phase1[x - window // 2:x + window // 2 + 1])

        corcof_P = ref1[max_idx] - ref1[min_idx]
        flag_P = -1 if corcof_P < 0 else 1

        tmp_I = mapminmax(flag_I * I, 0, 1)
        tmp_Q = mapminmax(flag_Q * Q, 0, 1)
        tmp_P = mapminmax(flag_P * ref1, 0, 1)

        record_I1[u, :] = tmp_I
        record_Q1[u, :] = tmp_Q
        record_P1[u, :] = tmp_P

        I = I_sig2 * np.cos(theta_prime) + Q_sig2 * np.sin(theta_prime)
        Q = -I_sig2 * np.sin(theta_prime) + Q_sig2 * np.cos(theta_prime)
        phase2 = np.unwrap(np.angle(flag_I * I + 1j * flag_Q * Q))
        phase2_1 = np.copy(phase2)
        for x in range(window // 2, len(phase2) - window // 2):
            phase2_1[x] = np.mean(phase2[x - window // 2:x + window // 2 + 1])

        tmp_I = mapminmax(flag_I * I, 0, 1)
        tmp_Q = mapminmax(flag_Q * Q, 0, 1)
        tmp_P = mapminmax(flag_P * ref2, 0, 1)

        record_I2[u, :] = tmp_I
        record_Q2[u, :] = tmp_Q
        record_P2[u, :] = tmp_P

    print('DualRotate end')
    return record_I1, record_Q1, record_P1, record_I2, record_Q2, record_P2


def DualRotate12(sig1, sig2, raw_sig1, raw_sig2, sample, breath_gt):
    print('DualRotate start')
    wid, len_ = sig1.shape
    record_I1 = np.zeros((6, wid, len_))  # 6 angles for I1
    record_Q1 = np.zeros((6, wid, len_))  # 6 angles for Q1
    record_P1 = np.zeros((6, wid, len_))  # 6 angles for P1
    record_I2 = np.zeros((6, wid, len_))  # 6 angles for I2
    record_Q2 = np.zeros((6, wid, len_))  # 6 angles for Q2
    record_P2 = np.zeros((6, wid, len_))  # 6 angles for P2

    for u in range(wid):
        cut1 = sig1[u, :]
        cut2 = sig2[u, :]
        I_sig1 = np.real(cut1)
        Q_sig1 = np.imag(cut1)
        I_sig2 = np.real(cut2)
        Q_sig2 = np.imag(cut2)
        ref = breath_gt[u, :]
        min_val, min_idx = min(ref[:200]), np.argmin(ref[:200])
        max_val, max_idx = max(ref[:200]), np.argmax(ref[:200])

        # Rotate in increments of π/6 from 0 to π (6 values: 0, π/6, π/3, π/2, 2π/3, 5π/6)
        angles = np.linspace(0, np.pi, 6)

        for i, angle in enumerate(angles):
            # Apply rotation for sig1
            I1_rot = I_sig1 * np.cos(angle) + Q_sig1 * np.sin(angle)
            Q1_rot = -I_sig1 * np.sin(angle) + Q_sig1 * np.cos(angle)
            I2_rot = I_sig2 * np.cos(angle) + Q_sig2 * np.sin(angle)
            Q2_rot = -I_sig2 * np.sin(angle) + Q_sig2 * np.cos(angle)

            ref1 = np.unwrap(np.angle((I1_rot + 1j * Q1_rot)))
            ref2 = np.unwrap(np.angle((I2_rot + 1j * Q2_rot)))

            corcof_I = I1_rot[max_idx] - I1_rot[min_idx]
            corcof_Q = Q1_rot[max_idx] - Q1_rot[min_idx]
            corcof_P = ref1[max_idx] - ref1[min_idx]

            flag_I = -1 if corcof_I < 0 else 1
            flag_Q = -1 if corcof_Q < 0 else 1
            flag_P = -1 if corcof_P < 0 else 1

            # Normalize and store the rotated results
            tmp_I1 = mapminmax(flag_I * I1_rot, 0, 1)
            tmp_Q1 = mapminmax(flag_Q * Q1_rot, 0, 1)
            tmp_P1 = mapminmax(flag_P * ref1, 0, 1)

            tmp_I2 = mapminmax(flag_I * I2_rot, 0, 1)
            tmp_Q2 = mapminmax(flag_Q * Q2_rot, 0, 1)
            tmp_P2 = mapminmax(flag_P * ref2, 0, 1)

            record_I1[i, u, :] = tmp_I1
            record_Q1[i, u, :] = tmp_Q1
            record_P1[i, u, :] = tmp_P1
            record_I2[i, u, :] = tmp_I2
            record_Q2[i, u, :] = tmp_Q2
            record_P2[i, u, :] = tmp_P2

    print('DualRotate end')
    return record_I1, record_Q1, record_P1, record_I2, record_Q2, record_P2
