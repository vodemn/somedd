import matplotlib.pyplot as plt
import numpy as np

from lab8 import run_lab

layer_size_range = np.linspace(10, 256, num=7)
lr_range = np.linspace(0.1, 1, num=19)
best_result = 0
best_params = (layer_size_range[0], lr_range[0])

for i, layer_size in enumerate(layer_size_range):
    ls = int(layer_size)
    layer_result = []
    for j, lr in enumerate(lr_range):
        result = run_lab(lr=lr, first_layer_size=ls, show_plot=False)
        layer_result.append(result)
        if result > best_result:
            best_result = result
            best_params = (ls, lr)
    sub = plt.subplot(1, layer_size_range.size, i + 1)
    sub.set_title(str(layer_size_range[i]))
    sub.stem(lr_range, layer_result)

print('Best accuracy:', best_result, 'with params:', best_params)
plt.show()

# Best accuracy: 0.9309090909090909 with params: (10, 0.4)