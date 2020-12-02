import numpy as np
import pystan
import matplotlib.pyplot as plt

N = 500
x = np.linspace(0, 1, N)

omega = 8 * np.pi
omega2 = 6 * np.pi
mu = 0.2
A = 1.5
B = 0.7
tau = 2.0
sigma = 0.1

f = mu + np.exp(-tau*x) *(A * np.sin(omega * x + omega2 * np.power(x,2)) + B * np.cos(omega * x + omega2 * np.power(x,2)))
y = f + sigma * np.random.randn(N)

plt.plot(x, f)
plt.plot(x, y, 'o')
plt.legend(['True function','measurements'])
plt.show()

# model_path = 'stan/arx_st.pkl'
# if Path(model_path).is_file():
#     model = pickle.load(open(model_path, 'rb'))
# else:
#     model = pystan.StanModel(file='stan/arx_st.stan')
#     with open(model_path, 'wb') as file:
#         pickle.dump(model, file)
model = pystan.StanModel(file='model.stan')

stan_data = {'N':N,
             'x':x,
             'y':y
             }

fit = model.sampling(data=stan_data, iter=2000, chains=4)

# extract the results
traces = fit.extract()

A_hmc = traces['A']
B_hmc = traces['B']
tau_hmc = traces['tau']
alpha_hmc = traces['alpha']
beta_hmc = traces['beta']
mu_hmc = traces['mu']
sig_e_hmc = traces['sig_e']

plt.subplot(3,3,1)
plt.hist(np.abs(A_hmc),30)
plt.axvline(A, linestyle='--', color='k')
plt.title('A')

plt.subplot(3,3,2)
plt.hist(np.abs(B_hmc),30)
plt.axvline(B, linestyle='--', color='k')
plt.title('B')

plt.subplot(3,3,3)
plt.hist(tau_hmc,30)
plt.axvline(tau, linestyle='--', color='k')
plt.title('tau')

plt.subplot(3,3,4)
plt.hist(alpha_hmc,30)
plt.axvline(omega, linestyle='--', color='k')
plt.title('alpha')

plt.subplot(3,3,5)
plt.hist(beta_hmc,30)
plt.axvline(omega2, linestyle='--', color='k')
plt.title('beta')

plt.subplot(3,3,6)
plt.hist(mu_hmc,30)
plt.axvline(mu, linestyle='--', color='k')
plt.title('mu')

plt.subplot(3,3,7)
plt.hist(sig_e_hmc)
plt.axvline(sigma, linestyle='--', color='k')
plt.title('sig_e')

plt.show()