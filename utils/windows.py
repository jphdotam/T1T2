import numpy as np

SEQUENCE_WINDOWS = {"T1w": [{'wc': 1300, 'ww': 1300}, {'wc': 500, 'ww': 1000}],
                    "T2w": [{'wc': 60, 'ww': 120}],
                    "PDw": None,
                    "T1": [{'wc': 1300, 'ww': 1300}, {'wc': 500, 'ww': 1000}],
                    "T2": [{'wc': 60, 'ww': 120}]}


def window_numpy(nparray, window_centre, window_width, cmap=None, rescale_255=True):
    v_min = window_centre - (window_width // 2)
    v_max = window_centre + (window_width // 2)
    nparray = np.clip(nparray, v_min, v_max)
    nparray = nparray - v_min
    nparray = nparray / window_width
    if rescale_255:
        nparray = (nparray * 255).astype(np.uint8)
    if cmap:
        nparray = cmap(nparray)
    return nparray
