""" Algorithm to calculate the Polynomial Chirp Transformation parameters as 
    described by:
	Polynomial Chirplet Transform With Application to Instantaneous
	Frequency Estimation Z. K. Peng, G. Meng, F. L. Chu, Z. Q. Lang, W. M.
	Zhang, and Y. Yang
"""

import numpy as np
import matplotlib.pyplot as plt
from math import inf
from scipy.integrate import romb
from scipy.signal import convolve, stft, get_window, hilbert
from scipy.signal.windows import gaussian
from numpy.polynomial.polynomial import Polynomial, polyval
import matplotlib.cm as cm

#  complete
def GaussianWindow(times, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (times / sigma) ** 2)

def TesterFunc(data, alpha):
    sum_ = np.zeros(data.size)
    for i in range(alpha.size):
        sum_ = sum_ + alpha[i] * data ** i
    return sum_


class PolynomialCT:
    """  Calculate the Polynomial Chirp Transformation parameters

        ===== Attributes =======
        alpha: polynomial coefficients from highest to lowest order
        z: signal data
        fs: sampling rate
        win_size: Size of sampling window

        hidden:
        _polynomial: underlying polynomial funtion
        tfd: current estimate of the polynomial chirplet
    """
    # not
    def __init__(self, data, fs=2*np.pi, win_size=256, overlap=None,
                 poly_order=6, initial_alpha=None):
        """ Initialize class. 
        """ 
        if initial_alpha is None:
            alpha = np.zeros(poly_order + 1)
        else:
            alpha = initial_alpha
        self.alpha = alpha

        self.z = data
        self.fs = fs
        self.poly_order = poly_order
        self._polynomial = Polynomial(alpha)  
        # transform frequency
        num_time_bins = self.z.size // win_size
        num_freq_bins = win_size // 2
        self.win_size = win_size
        if overlap is None:
            self.overlap = num_freq_bins // 2
        else:
            self.overlap = overlap
        self.tfd = np.zeros((num_freq_bins, num_time_bins), dtype='complex')

    # complete
    def _FRotationOperator(self, times):
        """ Calculate the nonlinear Frequency Rotation operator for self.z, 
            using parameters self.alpha for times <times>.

            Output: (times.size,) numpy.ndarray
        """
        sum_ = np.zeros(times.size, dtype='complex')
        for i in range(2, self._polynomial.degree() + 2):
            sum_ = sum_ + 1/ i * self.alpha[i-1] * times ** i
        return np.exp(-1j * sum_)

    # complete
    def _FShiftOperator(self, times, t0):
        """ Calculate the nonlinear Frequency Shift operator for self.z, 
            using parameters self.alpha for times <times>.

            Output: (num_time_bins, win_size) numpy.ndarray
        """
        if (t0 - self.win_size // 2) >= 0:
            time_win = times[t0 - self.win_size // 2: t0 + self.win_size // 2]
        else:
            time_win = times[:t0 + self.win_size // 2]

        sum_ = np.zeros(time_win.shape, dtype='complex')
        mid_value = times[t0]
        for k in range(2, self._polynomial.degree() + 2):
            sum_ = sum_ + self.alpha[k-1] * time_win ** (k-1) * mid_value
        return np.exp(1j * sum_)
    
    def _STFT(self):
        """ Helper function for calculating the PCT. This function does all the
            heavy lifting. Output the PCT(t, k) where k is the frequency index.
            The STFT is computed with no overlap and a gaussian window. 

            Output: (num_freq_bins, num_time_bins) numpy.ndarray
        """
        num_time_bins = self.z.size // (self.win_size - self.overlap) #+ 1
        num_freq_bins = self.win_size // 2
        num_stfts = self.win_size // (self.win_size - self.overlap)
        Zxx = np.zeros((num_freq_bins, num_time_bins), dtype='complex')

        times = np.arange(self.z.size) / self.fs  # divide by sampling rate
        freq_rot = self._FRotationOperator(times)
        # gaussian window
        window = 1/(np.sqrt(2 * np.pi) * self.win_size // 2) *\
                 get_window(('gaussian', self.win_size // 2), self.win_size)  

        for m in range(num_stfts):
            for i in range(num_time_bins):
                t = i * (self.win_size - self.overlap)
                if (t + self.win_size) > self.z.size:
                    continue
                if t - self.win_size // 2 < 0:
                    x = self.z[:t + self.win_size // 2 ] *\
                            freq_rot[:t + self.win_size //2]
                    # further transform the signal with the correct shift operator
                    freq_shift = self._FShiftOperator(times, t)
                    x_transform = x * window[:x.size] * freq_shift
                else: 
                    x = self.z[t - self.win_size // 2:t + self.win_size // 2 ] *\
                            freq_rot[t - self.win_size // 2:t + self.win_size //2]
                    # further transform the signal with the correct shift operator
                    freq_shift = self._FShiftOperator(times, t)
                    x_transform = x * window[:x.size] * freq_shift

                output = np.fft.fft(x_transform)[:num_freq_bins]
                factor = 1 - m * (self.win_size - self.overlap)
                assert(factor <= 1)
                Zxx[:, i] = Zxx[:, i] + factor * np.power(output, 2)
        Zxx = np.sqrt(Zxx) / (self.win_size - self.overlap)

        t = np.arange(0,num_time_bins) * ((self.win_size - self.overlap) / self.fs)
        f = np.arange(0,num_freq_bins) * (self.fs / self.win_size)
        # plt.pcolormesh(t,f,np.abs(Zxx), cmap=cm.get_cmap('jet'),shading='gouraud')
        # plt.ylabel('freq')
        # plt.show()
        return Zxx

    # complete
    def REN(self, pct):
        # frequency correction due to infinite integral
        integrand = np.log10(np.power(np.abs(pct), 3))
        return -1 * romb(romb(integrand, axis=1), axis=0)

    def _update_polynomial(self):
        # generate a line
        # print(self.alpha)
        num_time_bins = self.z.size // (self.win_size - self.overlap) # + 1
        num_freq_bins = self.win_size // 2
        # scale the bins
        freq_values = np.arange(0,num_freq_bins) *\
                      (self.fs / self.win_size) 
        tfd_peak = freq_values[np.argmax(np.abs(self.tfd), axis=0)]
        time_values = np.linspace(0,num_time_bins,tfd_peak.size) *\
                      ((self.win_size - self.overlap) / self.fs)
        fitted_polynomial = self._polynomial.fit(time_values[1:-1],
                                                 tfd_peak[1:-1],
                                                 self.poly_order)
        self._polynomial = fitted_polynomial.convert().copy()
        self.alpha = self._polynomial.coef
        # plt.plot(time_values[1:], tfd_peak[1:], 'r.')
        # plt.plot(time_values[1:], polyval(time_values[1:], self.alpha, 'k'))
        # plt.show()

    def PCT(self, initial_state=False):
        """ Calculate the polynomial chirp transformation using the parameters 
            <alpha> for  a analytic signal z[n] <z>.

            This function updates the following attributes: polynomial, ,
            tfd, and alpha
            === params ===
            initial_state: If True the polynomial parameters wont be estimated. 
        """
        # update polynomial
        # print("PCT running...")
        # recalculate the params of the instantaneous frequency
        if not initial_state: 
            self._update_polynomial()

        # transform signal and compute the STFT
        self.tfd = self._STFT()
        # print("PCT end.")

    # complete
    def Run(self, err, max_iter=100, iter_count=False):
        """ Run the system.

            === params ===
            err: stopping condition form 0 < err < 1
            max_iter: maximum number of runs
            iter_count: if True return the number of iterations required to
                        reach termination criteria
        """
        change_in_ren = inf  # Current change in Renyi entropy
        self.PCT(True)
        count = 0
        while (change_in_ren > err) and (max_iter > count):
            prev_pct = self.tfd.copy()
            self.PCT()
            curr_pct = self.tfd
            #REN_prev = self.REN(prev_pct)
            #REN_curr = self.REN(curr_pct)
            #change_in_ren = np.abs((REN_curr - REN_prev) / REN_curr) 
            count += 1

if __name__ == "__main__":
    # Test case
    def s(t):
        return np.sin(4*np.pi*(10*t + 5/4*t**2 + 1/9*t**3 - 1/160*t** 4))
    
    sample_freq = 200  # hz
    time = np.arange(0, 15, 1/400)
    # noise from a normal distribution with (mean, std) = (0, sqrt(3))
    np.random.seed(123)
    noise = np.random.normal(0, np.sqrt(3),time.size)
    discrete_signal = s(time) + noise
    signal = hilbert(discrete_signal)
    pctTest = PolynomialCT(signal, sample_freq, 516, 256, poly_order=3)
    pctTest.Run(0.001, 5)
    print(pctTest.alpha)
    print(pctTest.alpha/(2*np.pi))
