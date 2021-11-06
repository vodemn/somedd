import numpy as np

def zig_zag(a) -> np.array:
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    for i, val in enumerate(diags):
        if (i % 2 == 1):
            diags[i] = np.flip(diags[i])

    return np.concatenate(diags)
