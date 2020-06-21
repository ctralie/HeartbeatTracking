"""
Programmer: Chris Tralie
Purpose: To perform periodicity scoring on p53 time series 
data under different "gray" levels of radiation
"""
import numpy as np
from scipy import stats
import scipy.io as sio
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams as plot_dgms
from SlidingWindow import *


def Sw1PerS(xk, Win, dT, do_plot = False):
    """
    Sliding Window 1-Persistence Scoring (Sw1PerS)
    of a time series
    Parameters
    ----------
    xk: ndarray(N)
        A time series on which to run Sw1PerS
    Win: int
        Sliding window length
    dT: float
        Increment between sliding windows
    do_plot: boolean
        Plot the time series next to the diagrams
    """
    X = getSlidingWindow(xk, Win, 1, dT)
    X = normalizeWindows(X)

    # Perform TDA and save the maximum persistence
    # in H1 divided by sqrt(3)
    dgms = ripser(X, coeff=13)['dgms']
    h1 = dgms[1]
    score = 0
    if h1.size > 0:
        score = np.max(h1[:, 1]-h1[:, 0])/np.sqrt(3)

    if do_plot:
        plt.clf()
        plt.subplot(121)
        plt.plot(xk)
        plt.subplot(122)
        plot_dgms(dgms)
        plt.title("Sw1PerS Score = %.3g"%score)
    return score

def pitch_detection(xk, do_plot=False, mean_center=True, detrend_win=None):
    """
    Periodicity scoring via autocorrelation peak picking,
    normalizing to the squared magnitude of the signal
    (so-called "pitch detection")
    Parameters
    ----------
    xk: ndarray(N)
        A time series on which to run Sw1PerS
    do_plot: boolean
        Plot the time series next to the diagrams
    mean_center: boolean
        Whether to mean center the time series before autocorrelation
    detrend_win: int
        If specified, use this window length to do sliding window
        detrending
    Returns
    -------
    idx: int
        The chosen period (pitch), computed as a biproduct
    score: float
        The score between 0 and 1
    y: ndarray(N-1)
        Normalized autocorrelation
    x: ndarray(N)
        Signal on which autocorrelation was done (possibly detrended)
    """
    if detrend_win:
        x = detrend_timeseries(xk, detrend_win)
    else:
        x = xk
    if mean_center:
        x = x - np.mean(x)
    y = np.correlate(x, x, 'full') 
    y = y[y.size//2:]
    y /= y[0]
    idx = np.arange(1, len(y)-1)
    idx = idx[(y[idx-1] < y[idx])*(y[idx+1] < y[idx])]
    if idx.size > 0:
        idx = idx[np.argmax(y[idx])]
        score = y[idx]
    else:
        idx = 0
        score = 0
    if do_plot:
        plt.clf()
        cols = 2
        if detrend_win:
            cols = 3
        plt.subplot(1, cols, 1)
        plt.plot(xk)
        plt.title("Original Signal")
        if detrend_win:
            plt.subplot(132)
            plt.plot(x)
            plt.title("Detrended Signal")
        plt.subplot(1, cols, cols)
        plt.plot(y)
        plt.stem([idx], [score])
        plt.title("Pitch Score = %.3g"%score)
    return idx, score, y, x

def getPeriodicityDistribution(filename, outfilename, block_hop, block_len, score_fn, preprocess_fn = lambda x: x, truncate_len = 240, do_plot=False):
    """
    Parameters
    ----------
    filename: string
        Path to file where the time series are
    outfilename: string
        The path to which to save the scores
    block_hop: int
        Amount to increment between blocks
    block_len: int
        Length of each block in which a sliding window is performed.
        If none, take the entire time series
    score_fn: function (xk, do_plot) -> float
        A function for scoring a time series xk for periodicity on a
        scale from 0 to 1, with an option to plot some auxiliary information
    preprocess_fn: function ndarray(N) -> ndarray(N)
        A function to preprocess the time series
    truncate_len: int
        The maximum length of each time series
    do_plot: boolean
        Whether to save plots of persistence diagrams for each block
        for each time series
    Returns
    -------
    scores: ndarrray(NSignals, NBlocks)
        A list of periodicity scores, in the range [0, 1], over all
        blocks over all time series
    """
    res = np.loadtxt(filename)
    N = res.shape[0]
    scores = np.array([])
    if do_plot:
        plt.figure(figsize=(20, 6))
    for i in range(N): # Loop over all time series
        x = res[i, :]
        x = x[0:min(x.size, truncate_len)]
        if np.sum(x == -1) > 0:
            # This signal has missing data, so skip it
            continue
        x = preprocess_fn(x)
        i1 = 0
        scoresi = []
        if not block_len:
            block_len = x.size
        while i1+block_len <= x.size: # Loop over all blocks
            print("%i %i"%(i, i1))
            # Pull out block and do sliding window
            xk = np.array(x[i1:i1+block_len])
            score = score_fn(xk, do_plot)
            scoresi.append(score)
            if do_plot:
                plt.savefig("%i_%i.png"%(i, i1))
            i1 += block_hop
        if scores.size == 0:
            scores = np.NaN*np.ones((N, len(scoresi)))
        scores[i, :] = np.array(scoresi)
        if i%10 == 0:
            np.savetxt(outfilename, scores, fmt="%.4g")
    np.savetxt(outfilename, scores, fmt="%.4g")
    return np.array(scores)

def compute_all_scores():
    """
    Compute all scores for all blocks for all time series
    in 0Gy, 2Gy, 4Gy, 10Gy
    """
    block_hop = 1 # Maximum data augmentation is to take all blocks
    block_len = 48 # Take the block to be one day
    
    Win = 11 # This is around the period of p53
    dT = 0.2 # Make sure the windows are sampled finely enough
    score_fn_sw1pers = lambda xk, do_plot: Sw1PerS(xk, Win, dT, do_plot)
    score_fn_autocorrelation = lambda xk, do_plot: pitch_detection(xk, do_plot)[1]
    preprocess_detrend = lambda xk: detrend_timeseries(xk, Win)
    do_plot=False
    
    score_fn = score_fn_sw1pers
    for score_fn, preprocess_fn, score_name in zip([score_fn_autocorrelation, score_fn_sw1pers], [preprocess_detrend, lambda x: x], ["pitch", "tda"]):
        """
        for gray in [0, 2, 4, 10]:
            filename = "data/p53_%iGy.txt"%gray
            outfilename = "data/p53_%iGy_%s_scores.txt"%(gray, score_name)
            getPeriodicityDistribution(filename, outfilename, block_hop, block_len, score_fn, preprocess_fn, do_plot)
        """
        filename = "data_july_2019/p53_full.txt"
        outfilename = "data_july_2019/p53_%s_scores.txt"%score_name
        getPeriodicityDistribution(filename=filename, outfilename=outfilename, block_hop=block_hop, block_len=block_len, score_fn=score_fn, preprocess_fn=preprocess_fn, do_plot=do_plot)


if __name__ == '__main__':
    compute_all_scores()