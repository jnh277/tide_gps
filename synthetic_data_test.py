import numpy as np
import pystan
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
from pathlib import Path
from scipy.optimize import leastsq
from tqdm import tqdm
from preprocess_data import load_iprn, preprocess_SNR, load_tide_gauge_data

def sim_model_lin_freq(x, mu=0, amp=1.6, phase=np.pi/4, tau=1.5, alpha=110, beta=30, sig_e=0.3):
    sim_sig = mu + np.exp(-tau * x) * amp * np.sin(alpha * x + beta * x**2 + phase)
    sim_y = sim_sig + sig_e*np.random.randn(len(sim_sig))
    return sim_sig, sim_y

def sim_model_quad_freq(x, mu=0, amp=1.6, phase=np.pi/4, tau=1.5, alpha=105, beta=100, gamma=-135, sig_e=0.3):
    sim_sig = mu + np.exp(-tau * x) * amp * np.sin(alpha * x + beta * x**2 + gamma * x**3 + phase)
    sim_y = sim_sig + sig_e*np.random.randn(len(sim_sig))
    return sim_sig, sim_y

def sim_model_tide_gauge(x, h_tg, mu=0, amp=1.6, phase=np.pi/4, tau=1.5, sig_e=0.3, lambda_1=0.19029):
    freqs = h_tg * 4 * np.pi / lambda_1
    sim_sig = mu + np.exp(-tau * x) * amp * np.sin(freqs * x+ phase)
    sim_y = sim_sig + sig_e*np.random.randn(len(sim_sig))
    return sim_sig, sim_y, freqs



if __name__ == "__main__":
    lambda1 = 0.19029
    # iprns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    iprn = 2

    N_tests = 1
    # load tide gauge height information as interpolant
    # h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')
    h_tg_interp = load_tide_gauge_data('raw_data/tdg_no_header.txt')
    theta_est_store = np.ones((N_tests, 7))
    cov_est_store = np.ones((N_tests, 7))

    # mode = 1        # linear frequency
    mode = 2        # quadratic frequency
    # mode = 3        # tide gauge based frequency

    for test in tqdm(range(N_tests), desc= 'Running trials'):
        # load data for this iprn
        segments = load_iprn(iprn, 'raw_data/dean2650.20snr')
        alpha = 111
        beta = 30
        amp = 1.6
        phase = np.pi/4
        mu = 0.00
        sig_e = 0.01
        tau = 1.5
        gamma = 30

        # stuff

        for j, segment in enumerate(segments):
            x_tmp, _, t_tmp, _, _ = preprocess_SNR(segment, min_elevation=5, max_sin_el=0.3, remove_trend=True)
            if len(x_tmp) == 0:
                continue
            x = x_tmp
            t = t_tmp

            if mode == 1:
                true = np.array([mu, amp, phase, tau, alpha, beta])
                labels = ['mu', 'amp', 'phase', 'tau', 'alpha', 'beta']
                sim_sig, y = sim_model_lin_freq(x, tau=tau, mu=mu, alpha=alpha, beta=beta, amp=amp, phase=phase, sig_e=sig_e)
            elif mode ==2:
                alpha = 105
                beta = 100
                gamma = -135
                true = np.array([mu, amp, phase, tau, alpha, beta, gamma])
                labels = ['mu', 'amp', 'phase', 'tau', 'alpha', 'beta', "gamma"]
                sim_sig, y = sim_model_quad_freq(x, tau=tau, mu=mu, alpha=alpha, beta=beta, amp=amp, phase=phase, sig_e=sig_e, gamma=gamma)

            elif mode==3:
                h_tg = h_tg_interp(t)
                sim_sig, y, freqs = sim_model_tide_gauge(x,h_tg, tau=tau, mu=mu, amp=amp, phase=phase, sig_e=sig_e)

                # fit a linear line to the tide gauge based frequencies
                phi = np.zeros((len(t), 3))
                phi[:, 0] = 1.0
                phi[:, 1] = x
                phi[:, 2] = x ** 2
                vars = np.matmul(np.linalg.pinv(phi), freqs)
                alpha = vars[0]
                beta = vars[1]
                gamma = vars[2]

                # phi = np.zeros((len(t), 3))
                # phi[:, 0] = 1.0
                # phi[:, 1] = x
                # phi[:, 2] = x**2
                # vars = np.matmul(np.linalg.pinv(phi), freqs)
                # alpha = vars[0]
                # beta = vars[1]
                # gamma = vars[2]

                true = np.array([mu, amp, phase, tau, alpha, beta, gamma])
                labels = ['mu', 'amp', 'phase', 'tau', 'alpha', 'beta', 'gamma']


            N = len(y)

            # get initial estiamte of frequency
            omegas = np.linspace(0, 200, 600)
            A = np.zeros(omegas.shape)
            B = np.zeros(omegas.shape)
            for i, omega in enumerate(omegas):
                sx = np.sin(omega * x) * 2
                cx = np.cos(omega * x) * 2
                A[i] = (np.mean(sx * y))
                B[i] = (np.mean(cx * y))

            mag = np.sqrt(A ** 2 + B ** 2)

            ind = np.argmax(mag)
            omega_init = omegas[ind]
            A_init = np.round(A[ind] * 100) / 100
            B_init = np.round(B[ind] * 100) / 100
            phase_init = np.arctan2(B_init,A_init)

            # plt.plot(omegas, mag)
            # plt.title('Peak: omega = ' + str(omega_init) + ' A = ' + str(A_init) + ' B = ' + str(B_init))
            # plt.axvline(alpha, linestyle='--', color='k', linewidth=2)
            # plt.show()

            # # nonlinear least squares fit (linear in x freq)
            # optimize_func = lambda theta: sim_model_lin_freq(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5])[0] - y
            # theta, H_inv = leastsq(optimize_func, [0, mag[ind], phase_init, 1.0, omega_init, 0.0], full_output=True)[0:2]
            # f_est = sim_model_lin_freq(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5])[0]
            #
            # nonlinear least squares fit (quad in x freq)
            optimize_func = lambda theta: sim_model_quad_freq(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6])[0] - y
            theta, H_inv = leastsq(optimize_func, [0, mag[ind], phase_init, 1.0, omega_init, 0.0, 0.0], full_output=True)[0:2]
            f_est = sim_model_lin_freq(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6])[0]

            plt.plot(x, f_est)
            plt.plot(x, sim_sig)
            plt.show()

            cov_theta = H_inv * np.var(f_est - y)
            cov_est_store[test, :] = np.diag(cov_theta)
            theta_est_store[test, :] = theta

            # plt.plot(x, sim_sig)
            # plt.plot(x, y, '.')
            # plt.plot(x, f_est)
            # plt.show()


    for i in range(7):
        plt.subplot(3, 3, i+1)
        plt.hist(theta_est_store[:, i], bins=int(np.sqrt(N_tests)), density=True)
        plt.axvline(true[i], linestyle='--', color='k', linewidth=2)
        # plt.axvline(theta_est_store[:, i].mean()-2*np.sqrt(cov_est_store[:, i]).mean(), linestyle='--', color='r', linewidth=1)
        # plt.axvline(theta_est_store[:, i].mean()+2*np.sqrt(cov_est_store[:, i]).mean(), linestyle='--', color='r', linewidth=1)
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()