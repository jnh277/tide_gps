import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
from preprocess_data import load_iprn, preprocess_SNR, load_tide_gauge_data
from helpers import calc_MAP

if __name__ == "__main__":
    iprns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32]
    # load tide gauge height as interpolant
    h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')
    for iprn in tqdm(iprns, desc='plotting results'):
        segments = load_iprn(iprn, 'raw_data/dean2650.20snr')
        for j in range(0,10):
            path = 'results/iprn_'+str(iprn)+'segment'+str(j)+'.pkl'
            if Path(path).is_file():
                x, y, t, _, _ = preprocess_SNR(segments[j], min_elevation=2.5, max_sin_el=0.3, remove_trend=True)
                h_tg = h_tg_interp(t)
                with open(path, 'rb') as handle:
                    traces = pickle.load(handle)
                    h_hmc = traces['h']
                    h_mean = h_hmc.mean(axis=0)
                    h_map = np.zeros((len(y)))
                    for i in range(len(y)):
                        h_map[i] = calc_MAP(h_hmc[:, i])

                    m = h_hmc.shape[0]
                    n_plot = 400
                    inds = np.random.choice(m, n_plot)
                    h_std = h_hmc.std(axis=1).mean()
                    for ind in inds:
                        plt.plot(t, h_hmc[ind, :], linewidth=2, color='g', alpha=0.03)
                    plt.plot(t, h_mean, color='k')
                    plt.plot(t, h_map, color='blue')
                    plt.plot(t, h_tg, color='red', ls='--')

    plt.plot(t, h_hmc[0, :], linewidth=2, color='g', alpha=0.2, label='hmc samples')
    plt.plot(t, h_mean, color='k', label='mean')
    plt.plot(t, h_map, color='blue', label='MAP')
    plt.plot(t, h_tg, color='red', ls='--', label='tide gauge')
    plt.ylim([0.8, 2.8])
    plt.legend()
    plt.title('Estimated vs tide gauge')
    plt.ylabel('Height [m]')
    plt.xlabel('Time [hours]')
    plt.show()