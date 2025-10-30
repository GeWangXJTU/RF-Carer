import numpy as np
from scipy.optimize import root
from scipy.optimize import minimize


def paramfun(s, p):
    x_b1, y_b1, x_b2, y_b2, theta, alpha_x, alpha_y = s
    F = np.zeros(7)
    for i in range(4):
        F[i] = alpha_x * ((p[2, i] - x_b2) * np.cos(theta) + (p[3, i] - y_b2) * np.sin(theta)) + x_b1 - p[0, i]
    for i in range(3):
        F[4 + i] = alpha_y * ((p[2, i] - x_b2) * np.sin(theta) - (p[3, i] - y_b2) * np.cos(theta)) + y_b1 - p[1, i]
    return F


def calculate_initial_guess(sig, sig2):
    sx_b1 = np.min(np.real(sig))
    sy_b1 = np.min(np.imag(sig))
    sx_b2 = np.min(np.real(sig2))
    sy_b2 = np.min(np.imag(sig2))
    stheta = np.angle(sig[0]) - np.angle(sig2[0])
    salpha_x = (np.max(np.real(sig)) - np.min(np.real(sig))) / (np.max(np.real(sig2)) - np.min(np.real(sig2)))
    salpha_y = (np.max(np.imag(sig)) - np.min(np.imag(sig))) / (np.max(np.imag(sig2)) - np.min(np.imag(sig2)))
    return [sx_b1, sy_b1, sx_b2, sy_b2, stheta, salpha_x, salpha_y]


def transformVectorNew(cut_sample, sig, sig2):
    print('transformVectorNew start')
    win = 4
    step = 1
    s0 = []
    sig_length = len(sig)
    num_points = (sig_length - win) // step

    x_b1 = np.zeros(num_points)
    y_b1 = np.zeros(num_points)
    x_b2 = np.zeros(num_points)
    y_b2 = np.zeros(num_points)
    theta = np.zeros(num_points)
    alpha_x = np.zeros(num_points)
    alpha_y = np.zeros(num_points)

    count = 0

    for i in range(0, sig_length - win, step):
        p = np.vstack(
            (np.real(sig[i:i + win]), np.imag(sig[i:i + win]), np.real(sig2[i:i + win]), np.imag(sig2[i:i + win])))
        fun = lambda s: paramfun(s, p)

        if count == 0:
            sx_b1 = np.min(np.real(sig))
            sy_b1 = np.min(np.imag(sig))
            sx_b2 = np.min(np.real(sig2))
            sy_b2 = np.min(np.imag(sig2))

            if np.imag(sig2[i]) < 0:
                stheta = np.angle(sig[i]) - np.angle(sig2[i])
            else:
                stheta = np.angle(sig2[i]) - np.angle(sig[i])

            salpha_x = (np.max(np.real(sig)) - np.min(np.real(sig))) / (np.max(np.real(sig2)) - np.min(np.real(sig2)))
            salpha_y = (np.max(np.imag(sig)) - np.min(np.imag(sig))) / (np.max(np.imag(sig2)) - np.min(np.imag(sig2)))

            s0 = [sx_b1, sy_b1, sx_b2, sy_b2, stheta, salpha_x, salpha_y]
        else:
            s0 = s

        # s = fsolve(fun, np.array(s0), xtol=1e-6, maxfev=20000)
        s = root(fun, np.array(s0), method='lm').x  # 使用Levenberg-Marquardt算法
        if np.array(s).size != 0:
            x_b1[count] = s[0]
            y_b1[count] = s[1]
            x_b2[count] = s[2]
            y_b2[count] = s[3]
            theta[count] = s[4]
            alpha_x[count] = s[5]
            alpha_y[count] = s[6]
            count += 1
    # Part signals: algorithm 0
    count = 0
    seg_sig = sig
    seg_sig2 = sig2
    seg_phase = seg_sig  # Assuming seg_phase and seg_sig are equivalent
    seg_phase2 = seg_sig2
    final_sig1 = np.zeros((len(seg_sig) // cut_sample, cut_sample), dtype=complex)
    final_sig2 = np.zeros((len(seg_sig2) // cut_sample, cut_sample), dtype=complex)

    for v in range(0, len(seg_sig) - cut_sample + 1, cut_sample):
        compare_sig = seg_sig[v:v + cut_sample]
        compare_sig2 = seg_sig2[v:v + cut_sample]
        raw_phase = np.angle(seg_phase[v:v + cut_sample])
        raw_phase2 = np.angle(seg_phase2[v:v + cut_sample])

        residual = np.zeros(len(x_b1))
        trans_sig = np.zeros(cut_sample, dtype=complex)
        trans_sig2 = np.zeros(cut_sample, dtype=complex)

        for j in range(len(x_b1)):
            try_x_b1 = x_b1[j]
            try_y_b1 = y_b1[j]
            try_x_b2 = x_b2[j]
            try_y_b2 = y_b2[j]
            try_t = theta[j]
            try_ax = alpha_x[j]
            try_ay = alpha_y[j]

            tmp_sig = compare_sig - complex(try_x_b1, try_y_b1)
            tmp_sig2_x = try_ax * (np.real(compare_sig2) - try_x_b2) * np.cos(try_t) + try_ay * (
                    np.imag(compare_sig2) - try_y_b2) * np.sin(try_t)
            tmp_sig2_y = try_ay * (np.real(compare_sig2) - try_x_b2) * -np.sin(try_t) + try_ax * (
                    np.imag(compare_sig2) - try_y_b2) * np.cos(try_t)
            tmp_sig2 = tmp_sig2_x + 1j * tmp_sig2_y
            residual[j] = np.sum(np.abs(tmp_sig - tmp_sig2))

        idx = np.argmin(residual)
        min_x_b1 = x_b1[idx]
        min_y_b1 = y_b1[idx]
        min_x_b2 = x_b2[idx]
        min_y_b2 = y_b2[idx]
        min_t = theta[idx]
        min_ax = alpha_x[idx]
        min_ay = alpha_y[idx]

        trans_sig = compare_sig - complex(min_x_b1, min_y_b1)
        trans_sig2_x = min_ax * (
                (np.real(compare_sig2) - min_x_b2) * np.cos(min_t) + (np.imag(compare_sig2) - min_y_b2) * np.sin(
            min_t))
        trans_sig2_y = min_ay * (
                (np.real(compare_sig2) - min_x_b2) * -np.sin(min_t) + (np.imag(compare_sig2) - min_y_b2) * np.cos(
            min_t))
        trans_sig2 = trans_sig2_x + 1j * trans_sig2_y
        temp_center = complex(np.median(np.real(trans_sig)), np.median(np.imag(trans_sig)))
        trans_sig -= temp_center
        trans_sig2 -= temp_center
        final_sig1[count] = trans_sig
        final_sig2[count] = trans_sig2
        count += 1
    print('transformVectorNew end')
    return final_sig1, final_sig2


def transform_and_minimize(data1, data2):
    print('transformVectorNew start')
    data1_center = np.median(data1)
    data1 = data1 - data1_center

    def transformation(params, data2):
        angle, x_trans, y_trans, scale = params
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        data2_real = np.real(data2)
        data2_imag = np.imag(data2)
        points = np.vstack((data2_real, data2_imag))

        transformed_points = scale * rotation_matrix @ points

        transformed_points[0, :] += x_trans
        transformed_points[1, :] += y_trans

        return transformed_points[0, :] + 1j * transformed_points[1, :]

    def average_distance(params):
        transformed_data2 = transformation(params, data2)
        distances = np.abs(data1 - transformed_data2)
        return np.mean(distances)

    initial_params = [0, 0, 0, 1]

    bounds = [(None, None), (None, None), (None, None), (1e-6, None)]

    result = minimize(average_distance, np.array(initial_params), bounds=bounds)
    """
    Nelder-Mead (不使用梯度)
    Powell (方向集法)
    CG (共轭梯度法)
    BFGS (准牛顿法，无边界)
    L-BFGS-B (默认，带边界约束的准牛顿法)
    TNC (信赖域牛顿共轭梯度法)
    COBYLA (约束优化)
    SLSQP (序列二次规划)
    """
    best_params = result.x
    transformed_data2 = transformation(best_params, data2)
    # data2_center = np.median(transformed_data2)
    # transformed_data2 = transformed_data2 - data2_center
    min_avg_distance = result.fun

    data1 = data1.reshape(1, -1)
    transformed_data2 = transformed_data2.reshape(1, -1)
    print('transformVectorNew end')
    return best_params, data1, transformed_data2, min_avg_distance
