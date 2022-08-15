""" Short functions for data-preprocessing and data-loading. """

import numpy as np
import cv2
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def sample(a, len):
    """Samples a sequence into specific length."""
    return np.interp(
        np.linspace(
            1, a.shape[0], len), np.linspace(
            1, a.shape[0], a.shape[0]), a)


def detrend(signal, Lambda):
    signal_length = signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def label_fft(predictions, labels, signal='pulse', fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2],
                        btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

    if signal == 'pulse':
        pred_window = detrend(np.cumsum(predictions), 100)
        label_window = detrend(np.cumsum(labels), 100)
    else:
        pred_window = np.cumsum(predictions)

    if bpFlag:
        pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))
        label_window = scipy.signal.filtfilt(b, a, np.double(label_window))

    pred_window = np.expand_dims(pred_window, 0)
    label_window = np.expand_dims(label_window, 0)
    # Predictions FFT
    N = next_power_of_2(pred_window.shape[1])
    f_prd, pxx_pred = scipy.signal.periodogram(
        pred_window, fs=fs, nfft=N, detrend=False)
    if signal == 'pulse':
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))
    else:
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))
    pred_window = np.take(f_prd, fmask_pred)
    # Labels FFT
    f_label, pxx_label = scipy.signal.periodogram(
        label_window, fs=fs, nfft=N, detrend=False)
    print("f_label",f_label.shape)
    print("pxx_label",pxx_label.shape)
    plt.plot(f_label*60, pxx_label.reshape(-1))
    plt.title("SCAMPS frequency domain:")
    plt.show()
    if signal == 'pulse':
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))
    else:
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))
    print(f_label)
    label_window = np.take(f_label, fmask_label)
    print(label_window)
    amp_window = np.take(pxx_label, fmask_label)
    print("label_window",label_window.shape)
    print("label_window",amp_window.shape)
    plt.plot(label_window.reshape(-1)*60, amp_window.reshape(-1))
    plt.title("SCAMPS frequency domain butter:")
    plt.show()
    # MAE
    temp_HR, temp_HR_0 = calculate_HR(
        pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)
    # temp_SNR = calculate_SNR(pxx_pred, f_prd, temp_HR_0, signal)

    return temp_HR_0, temp_HR