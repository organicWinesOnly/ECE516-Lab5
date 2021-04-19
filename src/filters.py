""" Filter the signal.
"""
from scipy.signal import *
import numpy as np
import matplotlib.pyplot as plt
from math import inf

# data = np.loadtxt("../data/.csv", delimiter=",")
                  #skiprows=start, max_rows=12000)

# =============================================================================
# Build butterworth filer (bandpass)
# second order butterworth lowpass filter with 40 Hz cutoff
# =============================================================================
#fs = 268  # sampling rate in Hz

def lowpass_filter(signal, fs, limit=40,
                   passband_ripple=3.0, stopband_att=60.0):
    """ Filter signal using a low pass Butterworth filter. <fs> sampling rate of
        the <signal>. 

        === Params ====
        Limit - is the lower bound of the filter in Hz
        passband_ripple - pass band attenuation
        stopband_att - stop band attenuation
    """
    nyq_limit = 0.5 * fs
    lowpass_cutoff = 40 / nyq_limit


    # Calculate the correct order of the filter
    order, wn = buttord(lowpass_cutoff, 0.05 + lowpass_cutoff,
                        gpass=passband_ripple,
                        gstop=stopband_att)

    sos = butter(order, wn,
                  btype='lowpass',
                  output='sos')

    filtered_signal = sosfilt(sos, signal)
    return filtered_signal
