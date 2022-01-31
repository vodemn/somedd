import matplotlib.pyplot as plt
import numpy as np
import scipy.io

m = scipy.io.loadmat('lab8/dataset.mat')
np.savetxt('inputs.csv', m['inputs'], delimiter=',')
np.savetxt('targets.csv', m['targets'], delimiter=',')