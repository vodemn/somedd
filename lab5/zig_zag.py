import numpy as np

def zig_zag(img: np.ndarray) -> np.ndarray:
    img = np.flip(img, axis=0)
    diags = [img.diagonal(i) if i % 2 == 0 
                else np.flip(img.diagonal(i))
                    for i in range(-img.shape[0] + 1, img.shape[1])]
    return np.concatenate(diags)
