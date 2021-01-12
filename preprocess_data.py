import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import leastsq

def load_iprn(iprn, filename):
    column_names = ['iprn', 'elevation', 'azimuth', 'time', '_5', '_6', 'SNR', '_8', '_9']
    df = pd.read_table(filename, delim_whitespace=True, header=None)
    df.columns = column_names
    # df = pd.read_csv('raw_data/dean2650.20snr', sep="  ", header=None)
    # df.columns(['iprn', '-', 'elevation'])

    min_samples = 300

    df2 = df.loc[df['iprn']==iprn].reset_index()
    N = len(df2)

    # work out inds to split based on time differences
    time_diff = np.diff(df2['time'].to_numpy())
    split_inds = [0]+list(np.where(time_diff > 300)[0]+1) + [N]

    # work out inds to split based on elevation rate direction changes
    el_rate_changes = np.diff(np.sign(np.diff(df2['elevation'].to_numpy())))
    split_inds += list(np.where(abs(el_rate_changes)>0)[0]+2)
    split_inds = list(set(split_inds))      # remove duplicates and sort
    split_inds.sort()

    # split whenever time difference greater than 300 secs and whenever the elevation rate changes sign
    segments = [df2.iloc[split_inds[n]:split_inds[n+1]].reset_index() for n in range(len(split_inds)-1)]

    # remove sequences with less than min_samples
    count = 0
    for i in range(len(segments)):
        if len(segments[i-count]) < min_samples:
            del segments[i-count]
            count += 1
    return segments

def preprocess_SNR(segment, max_sin_el=0.5, min_elevation=0.0, remove_trend=True):
    elevation = segment['elevation'].to_numpy()
    snr = segment['SNR'].to_numpy()
    azimuth = segment['azimuth'].to_numpy()
    x = np.sin(elevation / 180 * np.pi)
    y = np.power(10, snr / 20)
    t = segment['time'].to_numpy() / 3600      # convert to hours

    # remove any with original SNR < 10, make sure elevation is above horizon
    inds = (elevation > min_elevation) & (x < max_sin_el) & (snr > 10) & ((azimuth > 270) & (azimuth < 360))

    x = x[inds]
    y = y[inds]
    t = t[inds]
    n = len(x)

    if remove_trend:
        tmp = np.atleast_2d(x).T
        A = np.hstack([np.ones(tmp.shape), tmp, tmp * tmp, tmp * tmp * tmp])
        theta = np.matmul(np.linalg.pinv(A), y)
        y = y - np.matmul(A, theta)

    return x, y, t, n, inds

def load_tide_gauge_data(filename, h0=1.85, t0=21, t=None):
    df = pd.read_table(filename, delim_whitespace=True, header=None)
    t_tg = (df[0] + df[3]/24 + df[4]/24/60 - 10/24 + 18/86400 - t0).to_numpy() * 24 # hours into september 2020
    h = (h0-df[6])      # height in meters
    h_tg_interp = interpolate.interp1d(t_tg, h)
    return h_tg_interp

if __name__ == "__main__":
    iprn = 2
    segments = load_iprn(iprn, 'raw_data/dean2650.20snr')
    # min elevation examples 1, 5, 8

    for j, segment in enumerate(segments):
        x, y, t, n, inds = preprocess_SNR(segment, min_elevation=5, max_sin_el=0.4, remove_trend=True)
        if n > 300:
            # plt.subplot(2,1,1)
            # plt.plot(x, y,'.')
            # plt.xlabel('sin(elevation)')
            # plt.ylabel('SNR')
            # plt.title(str(j))
            # plt.show()
            plt.plot(np.arcsin(x)*180/np.pi, y,'.')
            # plt.axvline(30.0,ls='--',color='red')
            plt.xlabel('elevation [degrees]')
            plt.ylabel('SNR [volt/volt]')
            plt.title('Post trend removal')
            # plt.title('iprn: '+str(iprn)+' segment: '+str(j))
            plt.show()