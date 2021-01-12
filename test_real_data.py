import numpy as np
import pystan
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from preprocess_data import load_iprn, preprocess_SNR, load_tide_gauge_data
from helpers import calc_MAP

# iprn = 4
# j = 1
# iprn = 5
# j = 0
iprn = 8
j = 2
# iprn = 15
# j = 1
segments = load_iprn(iprn, 'raw_data/dean2650.20snr')
x, y, t, _, _ = preprocess_SNR(segments[j], min_elevation=2.5, max_sin_el=0.3, remove_trend=True)

lambda1 = 0.19029

h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')

# normalise measurements
mean_y = y.mean()
std_y = y.std()
y = (y-mean_y) / std_y

N = y.shape[0]

# get initial estiamte of frequency
omegas = np.linspace(0,200,600)
A = np.zeros(omegas.shape)
B = np.zeros(omegas.shape)
for i, omega in enumerate(omegas):
    sx = np.sin(omega*x)/np.sqrt(2)
    cx = np.cos(omega*x)/np.sqrt(2)
    A[i] = (np.sum(sx * y))
    B[i] = (np.sum(cx * y))

mag = np.sqrt(A**2+B**2)
ind = np.argmax(mag)
omega_init = omegas[ind]

A_init = np.round(A[ind]/N * 100)/100
B_init = np.round(B[ind]/N * 100)/100

plt.plot(omegas,mag)
plt.title('Peak: omega = '+str(omega_init)+' A = '+str(A_init)+' B = '+str(B_init))
plt.show()




model_path = 'model.pkl'
if Path(model_path).is_file():
    model = pickle.load(open(model_path, 'rb'))
else:
    model = pystan.StanModel(file=model_path[:-4]+'.stan')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
# model = pystan.StanModel(file='model.stan')

def init_function():
    output = dict(alpha=omega_init * np.random.uniform(0.8, 1.2)
                  )
    return output

stan_data = {'N':N,
             'x':x,
             'y':y,
             'lambda1':lambda1
             }

fit = model.sampling(data=stan_data, init=init_function, iter=6000, warmup=4000, chains=4)#, control=dict(adapt_delta=0.9, max_treedepth=13))

# extract the results
traces = fit.extract()

A_hmc = traces['A']
B_hmc = traces['B']
tau_hmc = traces['tau']
alpha_hmc = traces['alpha']
beta_hmc = traces['beta']
mu_hmc = traces['mu']
sig_e_hmc = traces['sig_e']
f_hmc = traces['f']
nu_hmc = traces['nu']
# sig_lin = traces['sig_lin']
h_hmc = traces['h']
# gamma_hmc = traces['gamma']

plt.subplot(3,3,1)
plt.hist(np.abs(A_hmc),30, density=True)
plt.title('A')

plt.subplot(3,3,2)
plt.hist(np.abs(B_hmc),30, density=True)
plt.title('B')

plt.subplot(3,3,3)
plt.hist(tau_hmc,30, density=True)
plt.title('tau')

plt.subplot(3,3,4)
plt.hist(alpha_hmc,30, density=True)
plt.title('alpha')

plt.subplot(3,3,5)
plt.hist(beta_hmc,30, density=True)
plt.title('beta')

plt.subplot(3,3,6)
plt.hist(mu_hmc,30, density=True)
plt.title('mu')

plt.subplot(3,3,7)
plt.hist(sig_e_hmc, density=True)
plt.title('sig_e')

plt.subplot(3,3,8)
plt.hist(nu_hmc, density=True)
plt.title('nu_hmc')

# plt.subplot(3,3,9)
# plt.hist(gamma_hmc, density=True)
# plt.title('gamma')

plt.tight_layout()

plt.show()

# plt.hist(sig_lin, 30, density=True)
# plt.title('Linear noise dependence')
# plt.show()

# rad/sec to HZ
# 1 Hz = 1/(2pi) rad/sec

plt.plot(x, y, '.')
plt.plot(x,f_hmc.mean(axis=0))
plt.title('signal fit')
plt.show()


h_map = np.zeros((N))
for i in range(N):
    h_map[i] = calc_MAP(h_hmc[:,i])

h_mean = h_hmc.mean(axis=0)

m = h_hmc.shape[0]
n_plot = 400
inds = np.random.choice(m, n_plot)

plt.plot(t, h_hmc[0,:], linewidth=2, color='g', alpha=0.2, label='hmc samples')
for ind in inds:
    plt.plot(t, h_hmc[ind,:], linewidth=2, color='g', alpha=0.03)
plt.plot(t, h_mean, color='k', label='mean')
plt.plot(t, h_map, color='blue', label='MAP')
plt.plot(t, h_tg_interp(t), color='red', ls='--', label='tide gauge')
plt.title('tide height')
plt.legend()
plt.xlabel('time')
plt.ylabel('height (m)')
plt.show()

