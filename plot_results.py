import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pickle
from tqdm import tqdm
from pathlib import Path


if __name__ == "__main__":

    js = []
    iprns = []
    stds = []
    means = []
    good_batches = []
    good_means = []
    for batch in tqdm(range(52), desc='plotting results'):
        # load batch
        data = loadmat('data/batch_'+str(batch)+'.mat')

        t = data['ti'][:, 0]
        x = data['xi'][:, 0]

        sel = x < 0.35
        if np.sum(sel) < 400:       # some minimum number of data points
            continue

        t = t[sel]
        h_tg = data['h_tg'][sel,0]

        path = 'results/batch_'+str(batch)+'.pkl'
        if Path(path).is_file():
            with open(path, 'rb') as handle:
                traces = pickle.load(handle)
                h_hmc = traces['h']
                h_mean = h_hmc.mean(axis=0)
                m = h_hmc.shape[0]
                n_plot = 400
                inds = np.random.choice(m, n_plot)

                # plt.plot(t, h_hmc[0, :], linewidth=2, color='g', alpha=0.2, label='hmc samples')
                # for ind in inds:
                #     plt.plot(t, h_hmc[ind, :], linewidth=2, color='g', alpha=0.01)
                # plt.plot(t, h_mean, color='k', label='mean')



                h_std = h_hmc.std(axis=0).mean()

                # if h_std < 0.01:
                if True:
                    # plt.plot(t.mean(), h_hmc.mean(), 'o', color='black')
                    plt.errorbar(t.mean(),h_hmc.mean(),2*h_std, color='black')
                    # for ind in inds:
                        # plt.plot(t, h_hmc[ind, :], linewidth=2, color='g', alpha=0.01)
                    # plt.plot(t, h_mean, color='k', label='mean')
                    good_batches.append(batch)
                    good_means.append(h_mean.mean())
                plt.plot(t, h_tg * 0.01, color='red', ls='--', label='tide gauge')


                iprns.append(data['iprn'])
                js.append(data['j'])
                stds.append(h_std)
                means.append(h_mean.mean())


    plt.show()