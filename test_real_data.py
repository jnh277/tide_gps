import numpy as np
import pystan
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
from pathlib import Path


# data = loadmat('test_data.mat')
data = loadmat('data/batch_1.mat')
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
    A[i] = (np.sum(sx * y))
    B[i] = (np.sum(cx * y))

mag = np.sqrt(A**2+B**2)
plt.plot(omegas,mag)
plt.show()

ind = np.argmax(mag)
omega_init = omegas[ind]


model_path = 'model2.pkl'
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

fit = model.sampling(data=stan_data, init=init_function, iter=2000, chains=4)

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
sig_lin = traces['sig_lin']
h_hmc = traces['h']
gamma_hmc = traces['gamma']

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

plt.subplot(3,3,9)
plt.hist(gamma_hmc, density=True)
plt.title('sig_lin')

plt.show()

# rad/sec to HZ
# 1 Hz = 1/(2pi) rad/sec

plt.plot(x, y, '.')
plt.plot(x,f_hmc.mean(axis=0))
plt.title('signal fit')
plt.show()

# h = f*lambda1/2
# lambda1 = 0.19029       # [m]

# h = lambda1 * alpha_hmc / (2 * np.pi) /2
# plt.hist(h, density=True)
# plt.title('tidal height estimate distribution')
# plt.xlabel('height [m]')
# plt.ylabel('p(h | y)')
# plt.show()
#
#
# hg = lambda1 * beta_hmc / (2 * np.pi) /2
# plt.hist(hg, density=True)
# plt.title('tidal height gradient estimate distribution')
# plt.xlabel('height gradient [m / sine(elevation)]')
# plt.ylabel('p(h | y)')
# plt.show()


h_mean = h_hmc.mean(axis=0)
# interesting observation, h might be linearly dependent on x,
# but because t is not a linear function of x,
# h is not linearly dependent on t

m = h_hmc.shape[0]
n_plot = 200
inds = np.random.choice(m, n_plot)

plt.plot(t, h_hmc[0,:], linewidth=2, color='g', alpha=0.2, label='hmc samples')
for ind in inds:
    plt.plot(t, h_hmc[ind,:], linewidth=2, color='g', alpha=0.01)
plt.plot(t, h_mean, color='k', label='mean')
plt.plot(t, data['h_tg'][sel,0]*0.01, color='red', ls='--', label='tide gauge')
plt.title('tide height')
plt.legend()
plt.xlabel('time')
plt.ylabel('height (m)')
plt.show()

