import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pickle
from tqdm import tqdm
from scipy.optimize import leastsq
from pathlib import Path
from preprocess_data import load_iprn, preprocess_SNR, load_tide_gauge_data

hs = []
ts = []
lambda1 = 0.19029
iprns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
         32]

# load tide gauge height information as interpolant
# h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')

h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')
hs = []
ts = []
for iprn in tqdm(iprns, desc='loading data'):
    # load data for this iprn
    segments = load_iprn(iprn, 'raw_data/dean2650.20snr')

    for j, segment in enumerate(segments):
        x, y, t, _, _ = preprocess_SNR(segment, min_elevation=2.5, max_sin_el=0.3, remove_trend=True)
        if len(x) == 0:
            continue
        h_tg = h_tg_interp(t)
        hs.append(h_tg)
        ts.append(t)

h_tg = np.hstack(hs)
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
