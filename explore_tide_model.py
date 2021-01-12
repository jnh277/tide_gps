import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pickle
from tqdm import tqdm
from scipy.optimize import leastsq
from pathlib import Path

hs = []
ts = []
for batch in tqdm(range(52), desc='loading data'):
    # load batch
    data = loadmat('data/batch_' + str(batch) + '.mat')

    t = data['ti'][:, 0]
    h_tg = data['h_tg'][:, 0]
    ts.append(t)
    hs.append(h_tg)

h_tg = np.hstack(hs)*0.01
t = np.hstack(ts)
ind = np.argsort(t)
t = t[ind]
h_tg = h_tg[ind]

optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] + x[4]*np.sin(x[5]*t+x[6]) - h_tg
est_amp, est_freq, est_phase, est_mean, est_amp2, est_freq2, est_phase2 = leastsq(optimize_func, [-0.615, 2*np.pi/12, 0.83, 1.6, 0.2, 2*np.pi/14, 0])[0]

# recreate the fitted curve using the optimized parameters
# est_amp = 1.2
# est_freq =
data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean + est_amp2*np.sin(est_freq2*t+est_phase2)



plt.plot(t,h_tg, color='r', label='tide gauge')
plt.plot(t, data_fit, label='model fit')
plt.show()
