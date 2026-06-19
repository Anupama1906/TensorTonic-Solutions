import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len==None:
        max_len = max(len(seq) for seq in seqs)
    X = np.full((len(seqs),max_len),pad_value)
    for i,row in enumerate(seqs):
        k = min(max_len,len(row))
        for y in range(k):
            X[i,y] = row[y]
    return X