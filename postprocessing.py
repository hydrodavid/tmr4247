import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.signal as sp
import scipy.integrate as si
import airywavelib as aw

def importDataFrame(datapath, infile):
    '''
    Import data in numpy format into pandas DataFrame
    :param infile: numpy bindary format (npz)
    :return: pandas DataFrame
    '''
    with np.load(os.path.join(datapath, infile), allow_pickle=True) as fp:

        data = fp['data'].tolist()
        colnames = fp['keys'].tolist()
        
    # Create pandas DataFrame from dict
    df = pd.DataFrame.from_dict(data, orient='columns')

    # Set column names from dataset
    column_names = dict(zip(df.keys(), colnames))  # This creates a dictionary with {oldname: newname}
    df.rename(columns=column_names, inplace=True)

    return df

def getPhaseDifference(tvec, signal1, signal2):
    '''
    Compute phase difference based on zero-upcrossings
    :param tvec: time vector
    :param signal1: first signal for comparison
    :param signal2: second signal for comparison
    :return: phase lag dt
    '''
    # Find indices of zero upcrossings
    iup1 = np.ravel(np.argwhere(np.diff(np.sign(signal1)) > 0))
    l1 = len(iup1)

    # Find indices of zero upcrossings
    iup2 = np.ravel(np.argwhere(np.diff(np.sign(signal2)) > 0))
    l2 = len(iup2)

    # Use the shortest array length
    lmin = np.amin([l1, l2])

    # compute time difference and take mean value
    dt = np.mean(tvec[iup2[:lmin]] - tvec[iup1[:lmin]])
    return dt

def getZeroUpcrossingPeriod(time, signal):
    '''
    Compute zero up-crossing period
    :param time: time vector
    :param signal1: first signal for comparison
    :param signal2: second signal for comparison
    :return: phase lag dt
    '''
    # Find indices of zero upcrossings
    iup = np.argwhere(np.diff(np.sign(signal)) > 0)
    numcross = len(iup)

    # Interpolate time for zero-crossing
    tzeros = time[iup] - signal[iup]*(time[iup] - time[iup-np.ones_like(iup)])/(signal[iup] - signal[iup-np.ones_like(iup)])
    Tper = np.mean(np.diff(tzeros, axis=0))

    return Tper


def getFilterCoeffs(fs, fcut=[], filtertype='low', order=3):
    '''
    Compute filter coefficients for a Butterworth digital filter of given order
    :param fs: sampling frequency [Hz]
    :param fcut: list of cut frequencies [Hz]
    :param filtertype: either low (pass), high (pass) or band (pass)
    :param order: filter order
    :return: b, a filter coefficients
    '''
    # Construct Butterworth filter
    nyq = 0.5*fs
    if filtertype=='band':
        assert len(fcut)==2, "filter cut-frequency mismatch for band pass filter"
        fc = [fcut[0]/nyq, fcut[1]/nyq]
    else:
        fc = fcut[0]/nyq
    b, a = sp.butter(order, fc, btype=filtertype)
    return b, a

def filterDataFrame(df, fcut, fs, filtertype='low'):

    # Construct a Butterworth low-pass filter with given cut-frequency
    b, a = getFilterCoeffs(fs, fcut, filtertype=filtertype)

    # Data columns to filter (exclude time vector)
    #datacolumns = [key for key in df.keys() if not key.startswith('Time')]

    # Filter data
    #df_lp_np = sp.filtfilt(b, a, df[datacolumns].iloc[:], axis=0)
    df_lp_np = sp.filtfilt(b, a, df.iloc[:], axis=0)

    # Create new DataFrame
    #df_lp = pd.DataFrame(df_lp_np, columns=datacolumns)
    df_lp = pd.DataFrame(df_lp_np, columns=df.keys())
    df_lp.insert(0, column='Time', value=df.index.values)
    df_lp.set_index("Time", inplace=True)

    return df_lp

def filterData(data, fcut, fs, filtertype='low'):

    # Construct a Butterworth low-pass filter with given cut-frequency
    b, a = getFilterCoeffs(fs, fcut, filtertype=filtertype)

    # Data columns to filter (exclude time vector)
    #datacolumns = [key for key in df.keys() if not key.startswith('Time')]

    # Filter data
    #df_lp_np = sp.filtfilt(b, a, df[datacolumns].iloc[:], axis=0)
    data_filtered = sp.filtfilt(b, a, data, axis=0)

    return data_filtered




if __name__=="__main__":

    datapath = r"C:\_work\ModelTestData\TMR4247\wavecal\numpy"
    datafile = "Wave8001_8011.npz"

    # Import data:
    df = importDataFrame(datapath, datafile)

    # Rename data columns with more convenient column names:
    #modkeys=['Time', 'eta3', 'F2', 'F3', 'acc3', 'WP2']
    modkeys=['Time', 'eta3', 'WP1', 'WP2', 'F2', 'F3', 'acc_rig', 'acc3']
    column_names = dict(zip(df.keys(), modkeys)) # dict{oldname: newname}
    df.rename(columns=column_names, inplace=True)

    # Measured signal is actuator stroke with positive direction downwards
    # We redefine positive direction to be upwards
    df["eta3"] = -df["eta3"]

    #print(df.head())

    # Sampling frequency
    dt = df["Time"].values[1] - df["Time"].values[0] # Time-delta between samples
    fs = round(1./dt)

    #Set the column named "Time" as the index
    df.set_index("Time", inplace=True)

    # Now we can use df.loc[starttime:stoptime] to extract time windows
    tstart = 29.5
    tstop = 34
    print(df.loc[tstart:tstop])

    # Filter data:
    fcut = 3 # [Hz] Cut frequency
    df_filtered = filterDataFrame(df, fcut, fs)

    # Get oscillation period
    Tper = getZeroUpcrossingPeriod(df.index.values, df_filtered['eta3'].values)
    omega = 2*np.pi/Tper

    # Corresponding wavenumber
    h = 1.0
    k = aw.findWaveNumber(omega, waterDepth=h)
    Cg = 0.5*omega/k*(1.0 + k*h/(np.sinh(k*h)*np.cosh(k*h)))
    modelpos = 7.
    reflectiontime = 2*modelpos/Cg
    print(reflectiontime)

    # Extract steady-state time-window
    df_sub = df_filtered.loc[tstart:tstop]
    #df_sub.set_index("Time", inplace=True)
    print(df_sub.head())

    # Estimate hydrodynamic force by subtracting inertia and restoring force
    rho = 1000.
    g = 9.81
    B = 0.5 # [m] Breadth
    L = 0.59 # [m] Length
    T = 0.15 # [m] Draft
    M = L*B*T*rho
    Aw = L*B # [m^2] waterplane area
    C33 = rho*g*Aw # [kg]
    Fhd = -df_sub['F3'].values + M*df_sub['acc3'].values + C33*df_sub['eta3'].values


    iups = np.ravel(np.argwhere(np.diff(np.sign(df_sub['eta3'].values)) < 0))

    # Number of oscillation periods
    nPer = len(iups) - 1
    print(iups)
    t0 = df_sub.index[iups[0]]

    # Heave oscillation amplitude
    eta3a = np.max(df_sub['eta3'].values)

    tvec = df_sub.index[iups[0]:iups[-1]]

    # Get phase difference between measured eta3 and cos(omega*t)
    dt = getPhaseDifference(tvec, df_sub['eta3'].iloc[iups[0]:iups[-1]], np.cos(omega*tvec))
    print(dt)

    # Get phase difference between measured eta3 and cos(omega*t)
    dt2 = getPhaseDifference(tvec, df_sub['acc3'].iloc[iups[0]:iups[-1]], -np.cos(omega*tvec))
    print(dt2)

    # phase angle:
    delta = dt2*omega

    # Verify phase angle
    fig0, ax0 = plt.subplots(1, 1)
    ax0.plot(tvec, df_sub['eta3'].iloc[iups[0]:iups[-1]], label=r'$\eta_3(t)$')
    ax0.plot(tvec, df_sub['acc3'].iloc[iups[0]:iups[-1]]/omega**2, label=r'$\ddot{\eta}_3(t)/\omega^2$')
    ax0.plot(tvec, eta3a*np.cos(omega*tvec + delta), label=r'$\eta_{3a}\cos(\omega t)$')
    ax0.legend()

    # Compute added mass and damping coefficients
    A33 = si.trapz(Fhd[iups[0]:iups[-1]] * np.cos(omega*tvec + delta), x=tvec) / (omega*eta3a*nPer*np.pi)
    B33 = si.trapz(Fhd[iups[0]:iups[-1]] * np.sin(omega*tvec + delta), x=tvec) / (eta3a*nPer*np.pi)
    print("omega={:.3f} rad/s, A33={:.3f} kg, B33={:.3f} kg/s".format(omega, A33, B33))
    print("omega_hat={:.3f} [-], Ca33={:.3f} [-], Cb33={:.3f} [-]".format(omega*np.sqrt(B/(2*g)), A33/M, B33/M*np.sqrt(B/(2*g))))

    # Plot time-series:
    fig, ax = plt.subplots(4, 1, sharex=True)
    df.plot(ax=ax[0], y='F3')
    ax[0].axvline(tstart, color='k')
    ax[0].axvline(tstop, color='k')
    ax[0].axvline(reflectiontime+24.5, color='r')
    df_filtered.plot(ax=ax[0], y='F3')
    ax[0].set_ylabel('F3 [N]')

    df.plot(ax=ax[1], y='acc3')
    df_filtered.plot(ax=ax[1], y='acc3')
    ax[1].axvline(tstart, color='k')
    ax[1].axvline(tstop, color='k')
    ax[1].set_ylabel('Acc. [m/s^2]')

    df.plot(ax=ax[2], y='eta3')
    df_filtered.plot(ax=ax[2], y='eta3')
    ax[2].axvline(tstart, color='k')
    ax[2].axvline(tstop, color='k')
    ax[2].set_ylabel('Heave [m]')

    df.plot(ax=ax[3], y='WP2')
    df_filtered.plot(ax=ax[3], y='WP2')
    ax[3].axvline(tstart, color='k')
    ax[3].axvline(tstop, color='k')
    ax[3].set_ylabel('WP [m]')
    ax[3].set_xlabel('Time [s]')

    plt.show()

