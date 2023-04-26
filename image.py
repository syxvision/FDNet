# -*- coding: utf-8 -*-
# @Time    : 2020/8/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import cv2
import numpy as np



def minmax(data_array: np.ndarray, up_bound: float = None) -> np.ndarray:
    """
    ::

        data_array = (data_array / up_bound)
        if min_value != max_value:
            data_array = (data_array - min_value) / (max_value - min_value)

    :param data_array:
    :param up_bound: if is not None, data_array will devided by it before the minmax ops.
    :return:
    """
    if up_bound is not None:
        data_array = data_array / up_bound
    max_value = data_array.max()
    min_value = data_array.min()
    if max_value != min_value:
        data_array = (data_array - min_value) / (max_value - min_value)
    return data_array

def read_gray_array(path, div_255=False, to_normalize=False, thr=-1, dtype=np.float32) -> np.ndarray:
    """
    1. read the binary image with the suffix `.jpg` or `.png`
        into a grayscale ndarray
    2. (to_normalize=True) rescale the ndarray to [0, 1]
    3. (thr >= 0) binarize the ndarray with `thr`
    4. return a gray ndarray (np.float32)
    """
    assert path.endswith(".jpg") or path.endswith(".png")
    assert not div_255 or not to_normalize
    gray_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert gray_array is not None, f"Image Not Found: {path}"

    if div_255:
        gray_array = gray_array / 255

    if to_normalize:
        gray_array = minmax(gray_array, up_bound=255)

    if thr >= 0:
        gray_array = gray_array > thr

    return gray_array.astype(dtype)


def read_color_array(path: str):
    assert path.endswith(".jpg") or path.endswith(".png")
    bgr_array = cv2.imread(path, cv2.IMREAD_COLOR)
    assert bgr_array is not None, f"Image Not Found: {path}"
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return rgb_array
