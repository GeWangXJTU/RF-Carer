import numpy as np


def scale_complex_array(arr, magnitude):
    if not np.iscomplexobj(arr):
        raise ValueError("输入数组必须包含复数类型的数据。")

    base = np.abs(arr)
    magnitude_factor = np.zeros_like(arr)

    non_zero_indices = base > 0
    magnitude_factor[non_zero_indices] = (base[non_zero_indices] - magnitude) / base[non_zero_indices]

    magnitude_factor[magnitude_factor < 0] = 0

    adjusted_arr = arr.copy()
    adjusted_arr[non_zero_indices] = magnitude_factor[non_zero_indices] * arr[non_zero_indices]
    return adjusted_arr


def rotate_scale_complex__array(arr, _phi, magnitude):
    if not np.iscomplexobj(arr):
        raise ValueError("输入数组必须包含复数类型的数据。")
    rotated_arr = arr * np.exp(-1j * _phi)

    base = np.abs(rotated_arr)
    magnitude_factor = np.zeros_like(arr)

    non_zero_indices = base > 0

    magnitude_factor = (base - magnitude) / base + 1e-16

    magnitude_factor = np.where(magnitude_factor < 0, np.abs(magnitude_factor), magnitude_factor)

    adjusted_arr = arr.copy()
    adjusted_arr = magnitude_factor * rotated_arr

    # 
    # non_zero_indices = base > 0  
    # magnitude_factor[non_zero_indices] = (base[non_zero_indices] - magnitude) / base[non_zero_indices]
    #
    # 
    # magnitude_factor[magnitude_factor < 0] = 0
    #
    # 
    # adjusted_arr = arr.copy()
    # adjusted_arr[non_zero_indices] = magnitude_factor[non_zero_indices] * rotated_arr[non_zero_indices]
    return adjusted_arr


def rotate_complex_numpy_array(arr, _phi):
    if not np.iscomplexobj(arr):
        raise ValueError("输入数组必须包含复数类型的数据。")
    rotated_arr = arr * np.exp(-1j * _phi)
    return rotated_arr
