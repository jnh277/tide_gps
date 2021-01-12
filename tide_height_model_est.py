import numpy as np
import pystan
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from tqdm import tqdm
from preprocess_data import load_iprn, preprocess_SNR, load_tide_gauge_data

if __name__ == "__main__":
    lambda1 = 0.19029
    iprns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    # load tide gauge height information as interpolant
    h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')
    ys = []
    ts = []
    xs = []
    Ns = []

    for iprn in tqdm(iprns, desc= 'load data'):
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

            xs.append(x)
            ys.append(y)
            ts.append(t)
            Ns.append(N)

        if len(xs) > 5:
            break

    sets = len(Ns)
    N_max = np.max(Ns)
    N_all = np.array(Ns)

    x_all = np.full(shape=(sets,N_max),fill_value=np.inf)
    y_all = np.full(shape=(sets, N_max), fill_value=np.inf)
    t_all = np.full(shape=(sets, N_max), fill_value=np.inf)

    # x_all = np.full(shape=(sets,N_max),fill_value=0.0)
    # y_all = np.full(shape=(sets, N_max), fill_value=0.0)
    # t_all = np.full(shape=(sets, N_max), fill_value=0.0)

    for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
        x_all[i, :N_all[i]] = x
        y_all[i, :N_all[i]] = y
        t_all[i, :N_all[i]] = t


    model_path = 'gps_tide_model.pkl'
    if Path(model_path).is_file():
        model = pickle.load(open(model_path, 'rb'))
    else:
        model = pystan.StanModel(file=model_path[:-4] + '.stan')
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    #
    #
    def init_function():
        output = dict(M2_freq=0.4559,
                      S2_freq=0.4482
                      )
        return output

    stan_data = {'sets': sets,
                 'max_N': N_max,
                 'N': N_all,
                 'x': x_all,
                 'y': y_all,
                 't': t_all,
                 'lambda1': lambda1
                 }
    print('Fitting the model')
    fit = model.sampling(data=stan_data, init=init_function, iter=6000, warmup=4000,
                         chains=4, control=dict( max_treedepth=11))  # , control=dict(adapt_delta=0.9, max_treedepth=13))
    #
    # extract the results
    traces = fit.extract()
    h = traces['h']
    s = 2
    h_tmp = h[:,s,:N_all[s]]
    # h_tmp = np.reshape(h[:, s, :N_all[s]],(-1))
    # inds = h_tmp < 2
    # h_tmp = h_tmp[inds]
    # h_tmp = np.reshape(h_tmp,(-1,N_all[s]))

    h_tg = h_tg_interp(t_all[s,:N_all[s]])

    plt.hist(h_tmp.mean(axis=1))
    plt.axvline(h_tg.mean())
    plt.show()

    plt.plot(t_all[s,:N_all[s]],h_tmp.mean(axis=0))
    plt.plot(t_all[s,:N_all[s]],h_tg)
    plt.show()

    #
        #
        #
