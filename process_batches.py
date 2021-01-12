import numpy as np
import pystan
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
from pathlib import Path
from tqdm import tqdm
from preprocess_data import load_iprn, preprocess_SNR, load_tide_gauge_data

if __name__ == "__main__":
    lambda1 = 0.19029
    iprns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    # load tide gauge height information as interpolant
    # h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')

    for iprn in tqdm(iprns, desc= 'processing GPS information'):
        # load data for this iprn
        segments = load_iprn(iprn, 'raw_data/dean2650.20snr')

        for j, segment in enumerate(segments):
            x, y, t, _, _ = preprocess_SNR(segment, min_elevation=2.5, max_sin_el=0.3, remove_trend=True)
            if len(x) == 0:
                continue

            # normalise measurements
            mean_y = y.mean()
            std_y = y.std()
            y = (y - mean_y) / std_y

            N = y.shape[0]

            # get initial estiamte of frequency
            omegas = np.linspace(0, 200, 600)
            A = np.zeros(omegas.shape)
            B = np.zeros(omegas.shape)
            for i, omega in enumerate(omegas):
                sx = np.sin(omega * x) / np.sqrt(2)
                cx = np.cos(omega * x) / np.sqrt(2)
                A[i] = (np.sum(sx * y))
                B[i] = (np.sum(cx * y))

            mag = np.sqrt(A ** 2 + B ** 2)

            ind = np.argmax(mag)
            omega_init = omegas[ind]

            model_path = 'model.pkl'
            if Path(model_path).is_file():
                model = pickle.load(open(model_path, 'rb'))
            else:
                model = pystan.StanModel(file=model_path[:-4] + '.stan')
                with open(model_path, 'wb') as file:
                    pickle.dump(model, file)


            def init_function():
                output = dict(alpha=omega_init * np.random.uniform(0.8, 1.2)
                              )
                return output


            stan_data = {'N': N,
                         'x': x,
                         'y': y,
                         'lambda1': lambda1
                         }
            fit = model.sampling(data=stan_data, init=init_function, iter=6000, warmup=4000,
                                 chains=4, refresh=6000)  # , control=dict(adapt_delta=0.9, max_treedepth=13))

            # extract the results
            traces = fit.extract()


            with open('results/iprn_'+str(iprn)+'segment'+str(j)+'.pkl', 'wb') as handle:
                pickle.dump(traces, handle, protocol=pickle.HIGHEST_PROTOCOL)
