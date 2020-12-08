import numpy as np
import pystan
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    lambda1 = 0.19029
    for batch in tqdm(range(52), desc='processing batches'):

        # load batch
        data = loadmat('data_restricted_270_360/batch_'+str(batch)+'.mat')

        y = data['yi'][:, 0]
        t = data['ti'][:, 0]
        x = data['xi'][:, 0]

        sel = x < 0.35
        if np.sum(sel) < 400:       # some minimum number of data points
            continue

        y = y[sel]
        x = x[sel]
        t = t[sel]

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
            cx = np.sin(omega * x) / np.sqrt(2)
            A[i] = (np.sum(sx * y))
            B[i] = (np.sum(cx * y))

        mag = np.sqrt(A ** 2 + B ** 2)
        # plt.plot(omegas,mag)
        # plt.show()

        ind = np.argmax(mag)
        omega_init = omegas[ind]

        model_path = 'model3.pkl'
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
        try:
            max_count = 5
            # traces_list = []
            # h_stds = []
            for c in range(max_count):
                fit = model.sampling(data=stan_data, init=init_function, iter=10000, warmup=7000, chains=4, refresh=5000)#, control=dict(adapt_delta=0.9, max_treedepth=13))

                # extract the results
                traces = fit.extract()
                h_hmc = traces['h']
                h_std = h_hmc.std(axis=0).mean()

                # traces_list.append(traces)
                # h_stds.append(h_std)
                if h_std < 0.01:
                    print('SUCCESS')
                    break
                else:
                    print('RETRY')

            # ind = np.argmin(h_stds)
            # traces = traces_list[ind]
            # traces['t'] = t
            # traces['h_tg'] = data['h_tg'][sel,0]
            with open('results3/batch_'+str(batch)+'.pkl', 'wb') as handle:
                pickle.dump(traces, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print('Error in fitting')

