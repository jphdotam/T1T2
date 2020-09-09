import os
import numpy as np


def pad_if_needed(img, min_height, min_width):
    input_height, input_width = img.shape[:2]
    new_shape = list(img.shape)
    new_shape[0] = max(input_height, min_height)
    new_shape[1] = max(input_width, min_width)
    row_from, col_from = 0, 0
    if input_height < min_height:
        row_from = (min_height - input_height) // 2
    if input_width < min_width:
        col_from = (min_width - input_width) // 2
    out = np.zeros(new_shape, dtype=img.dtype)
    out[row_from:row_from+input_height, col_from:col_from+input_width] = img
    return out


def center_crop(img, crop_height, crop_width, centre=None):
    """Either crops by the center of the image, or around a supplied point.
    Does not pad; if the supplied centre is towards the egde of the image, the padded
    area is shifted so crops start at 0 and only go up to the max row/col

    Returns both the new crop, and the top-left coords as a row,col tuple"""
    input_height, input_width = img.shape[:2]
    if centre is None:
        row_from = (input_height - crop_height)//2
        col_from = (input_width - crop_width)//2
    else:
        row_centre, col_centre = centre
        row_from = max(row_centre - (crop_height//2), 0)
        if (row_from + crop_height) > input_height:
            row_from -= (row_from + crop_height - input_height)
        col_from = max(col_centre - (crop_width//2), 0)
        if (col_from + crop_width) > input_width:
            col_from -= (col_from + crop_width - input_width)
    return img[row_from:row_from+crop_height, col_from:col_from+crop_width], (row_from, col_from)


def get_original_npy_path_from_exported_npz_path(npz_path, peter_dir):
    date, study, file, _end = os.path.basename(npz_path).split('__')
    peter_path = os.path.join(peter_dir, date, study, file + '.npy')
    return peter_path
