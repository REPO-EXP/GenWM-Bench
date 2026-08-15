
import numpy as np

def quantize(arr, min_val, max_val, levels, dtype=np.int64):
    
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(
            )
    if min_val >= max_val:
        raise ValueError(
            )

    arr = np.clip(arr, min_val, max_val) - min_val
    quantized_arr = np.minimum(
        np.floor(levels * arr / (max_val - min_val)).astype(dtype), levels - 1)

    return quantized_arr

def dequantize(arr, min_val, max_val, levels, dtype=np.float64):
    
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(
            )
    if min_val >= max_val:
        raise ValueError(
            )

    dequantized_arr = (arr + 0.5).astype(dtype) * (max_val -
                                                   min_val) / levels + min_val

    return dequantized_arr
