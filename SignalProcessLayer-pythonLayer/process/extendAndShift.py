import numpy as np


def extendAndShift(sequence, desired_length):
    current_length = len(sequence)
    if current_length >= desired_length:

        extended_sequence = sequence
    else:
        num_repeats = np.ceil(desired_length / current_length).astype(int)
        extended_sequence = np.tile(sequence, num_repeats)[:desired_length]

    shift_amount = (desired_length - current_length) % current_length

    extended_sequence = np.roll(extended_sequence, shift_amount)
    return extended_sequence



