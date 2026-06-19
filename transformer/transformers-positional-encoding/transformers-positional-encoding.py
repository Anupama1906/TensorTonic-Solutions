import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    P_list = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(d_model):
            if i&1:
                P_list[pos,i] = np.cos(pos/(10000**((i-1)/d_model)))
            else:
                P_list[pos,i] = np.sin(pos/(10000**((i)/d_model)))
    return P_list