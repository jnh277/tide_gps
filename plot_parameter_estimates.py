import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
from preprocess_data import load_iprn, preprocess_SNR, load_tide_gauge_data
from helpers import calc_MAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot estimates')
    parser.add_argument('-p', '--parameter', default='sig_e',
                        help='which parameter to plot')
    args, unk = parser.parse_known_args()
    parameter = args.parameter

    parameter_estimates = []

    iprns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32]
    # load tide gauge height as interpolant
    h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')
    iprn_plot = []
    j_plot = []
    h_tg_plot = []
    t_plot = []
    y_plot = []
    x_plot = []
    for iprn in tqdm(iprns, desc='loading results'):
        segments = load_iprn(iprn, 'raw_data/dean2650.20snr')
        for j in range(0,10):
            path = 'results/iprn_' + str(iprn) + 'segment' + str(j) + '.pkl'
            if Path(path).is_file():
                x, y, t, _, _ = preprocess_SNR(segments[j], min_elevation=2.5, max_sin_el=0.3, remove_trend=True)
                # h_tg = h_tg_interp(t)
                with open(path, 'rb') as handle:
                    traces = pickle.load(handle)
                    if parameter == 'h':
                        parameter_estimates.append(traces[parameter].mean(axis=1))
                        h_tg_plot.append(h_tg_interp(t).mean())
                    else:
                        parameter_estimates.append(traces[parameter])
                    iprn_plot.append(iprn)
                    j_plot.append(j)
                    t_plot.append(t)
                    x_plot.append(x)
                    y_plot.append((y-y.mean())/y.std())

    n = len(parameter_estimates)
    cols = 4
    rows = np.ceil(n/cols)
    for i in tqdm(range(n), desc='Plotting'):
        plt.subplot(int(rows), cols, i+1)
        if parameter=='f':
            n_plot = 800
            inds = np.random.choice(2000, n_plot)
            for ind in inds:
                plt.plot(x_plot[i], parameter_estimates[i][ind, :], linewidth=0.5, color='g', alpha=0.03)
            plt.plot(x_plot[i], y_plot[i], color='red', ls='--', linewidth=0.25)
        else:
            plt.hist(parameter_estimates[i], bins=30, density=True)
        if parameter=='h':
            plt.axvline(h_tg_plot[i], ls='--', color='k')
        # plt.ylabel('iprn: '+str(iprn_plot[i])+' j: '+str(j_plot[i]))
    plt.tight_layout()
    plt.show()