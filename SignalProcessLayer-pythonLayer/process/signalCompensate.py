import numpy as np

RANGE_RESOLUTION = 0.0514
FREQUENCY_CENTRAL = 7.29e9
BAND_WIDTH = 1.4e9
FREQUENCY_GENERATE = 40
LIGHT_SPEED = 2.998e8


def signal_compensation(signal, RANGE_RESOLUTION=0.0514, FREQUENCY_CENTRAL=7.29e9, LIGHT_SPEED=2.998e8, M=96):
    i = np.arange(1, M + 1)

    d = i * RANGE_RESOLUTION

    A_c = d ** 2

    theta_c = 2 * np.pi * FREQUENCY_CENTRAL * d / LIGHT_SPEED

    compensation_factor = A_c * np.exp(-1j * theta_c)

    compensated_signal = signal * compensation_factor[np.newaxis, :]

    return compensated_signal
