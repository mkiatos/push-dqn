import numpy as np


def min_max_scale(x, range, target_range, lib='np'):
    assert range[1] > range[0]
    assert target_range[1] > target_range[0]

    if lib == 'np' and isinstance(x, np.ndarray):
        range_min = range[0] * np.ones(x.shape)
        range_max = range[1] * np.ones(x.shape)
        target_min = target_range[0] * np.ones(x.shape)
        target_max = target_range[1] * np.ones(x.shape)
    else:
        range_min = range[0]
        range_max = range[1]
        target_min = target_range[0]
        target_max = target_range[1]

    return target_min + ((x - range_min) * (target_max - target_min)) / (range_max - range_min)


