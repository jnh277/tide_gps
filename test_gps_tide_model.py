import numpy as np
import pystan
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
from pathlib import Path


data = loadmat('test_data.mat')
# data = loadmat('data/batch_37.mat')       # max tide
# data = loadmat('data/batch_8.mat')          # min tide
lambda1 = 0.19029

y = data['yi'][:,0]
t = data['ti'][:,0]
x = data['xi'][:,0]

sel = x < 0.35
y = y[sel]
x = x[sel]
t = t[sel]


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
    cx = np.sin(omega*x)/np.sqrt(2)
    A[i] = (np.mean(sx * y))
    B[i] = (np.mean(cx * y))

mag = np.sqrt(A**2+B**2)
plt.plot(omegas,mag)
plt.show()

ind = np.argmax(mag)
omega_init = omegas[ind]
A_init = A[ind]
B_init = B[ind]


model_path = 'gps_tide_model.pkl'
if Path(model_path).is_file():
    model = pickle.load(open(model_path, 'rb'))
else:
    model = pystan.StanModel(file=model_path[:-4]+'.stan')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
# model = pystan.StanModel(file='model.stan')


def init_function():
    output = dict(
                  )
    return output

stan_data = {'N':N,
             'x':x,
             'y':y,
             'lambda1':lambda1,
             't':t
             }

fit = model.sampling(data=stan_data, init=init_function, iter=2000, chains=4)

# extract the results
traces = fit.extract()

A_hmc = traces['A']
B_hmc = traces['B']
tau_hmc = traces['tau']
# M1_freq_hmc = traces['M1_freq']
# M1_A_hmc = traces['M1_A']
# M1_B_hmc = traces['M1_B']
mu_hmc = traces['mu']
sig_e_hmc = traces['sig_e']
signal_hmc = traces['signal']
nu_hmc = traces['nu']
# sig_lin = traces['sig_lin']
h_hmc = traces['h']
# gamma_hmc = traces['gamma']
sf_hmc = traces['sf']

plt.subplot(3,3,1)
plt.hist(np.abs(A_hmc),30, density=True)
plt.title('A')

plt.subplot(3,3,2)
plt.hist(np.abs(B_hmc),30, density=True)
plt.title('B')

plt.subplot(3,3,3)
plt.hist(tau_hmc,30, density=True)
plt.title('tau')

# plt.subplot(3,3,4)
# plt.hist(M1_freq_hmc,30, density=True)
# plt.title('M1_freq')
# #
# plt.subplot(3,3,5)
# plt.hist(M1_A_hmc,30, density=True)
# plt.title('M1_A')
#
# plt.subplot(3,3,6)
# plt.hist(M1_B_hmc, density=True)
# plt.title('M1_B')

plt.subplot(3,3,7)
plt.hist(mu_hmc,30, density=True)
plt.title('mu')

plt.subplot(3,3,8)
plt.hist(sig_e_hmc, density=True)
plt.title('sig_e')


plt.subplot(3,3,9)
plt.hist(nu_hmc, density=True)
plt.title('nu_hmc')



plt.show()


plt.plot(x, y, '.')
plt.plot(x,signal_hmc.mean(axis=0))
plt.title('signal fit')
plt.show()





h_mean = h_hmc.mean(axis=0)

m = h_hmc.shape[0]
n_plot = 200
inds = np.random.choice(m, n_plot)

# plt.plot(t, h_hmc[0,:], linewidth=2, color='g', alpha=0.2, label='hmc samples')
# for ind in inds:
    # plt.plot(t, h_hmc[ind,:], linewidth=2, color='g', alpha=0.01)
plt.axhline(h_mean, color='k', label='mean')
plt.plot(t, data['h_tg'][sel,0]*0.01, color='red', ls='--', label='tide gauge')
plt.title('tide height')
plt.legend()
plt.xlabel('time')
plt.ylabel('height (m)')
plt.show()

