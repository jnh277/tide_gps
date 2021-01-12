from scipy.stats import gaussian_kde as kde
import numpy as np



def calc_MAP(x):
    min_x = np.min(x)
    max_x = np.max(x)
    pos = np.linspace(min_x, max_x, 100)
    kernel = kde(x)
    z = kernel(pos)
    return pos[np.argmax(z)]