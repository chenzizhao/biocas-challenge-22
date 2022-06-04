'''
for recording tasks: try preprocee_fft, preprocess_norm maybe (preprocee_stft,preprocess_mel,preprocess_wavelet)
for event tasks: try preprocee_stft,preprocess_mel,preprocess_wavelet
'''

from os import listdir
from os.path import join
from torchaudio import load
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
import librosa
from scipy.signal import butter, lfilter
#import librosa.display as display

WAV_DIR = "wav"
CLIP_DIR = "clip"
PROC_DIR = "processed"

def Normalization(x):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    for i in range(len(x)):
        
        x[i] = float(x[i]-min_x)/(max_x-min_x)
           
    return x

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def wavelet(sig):
    cA, out = pywt.dwt(sig, 'db8')
    cA, out = pywt.dwt(cA, 'db8')
    cA, out = pywt.dwt(cA, 'db8')
    A = cA
    
    for i in range(6):
        cA, cD = pywt.dwt(A, 'db8')
        A = cA
        out = np.hstack((out,cD))

    out = np.hstack((out,A))
        
    return out

def reshape(matrix):
    num = matrix.shape[0]
    length = math.ceil(np.sqrt(num))
    zero = np.zeros([np.square(length)-num,])
    matrix = np.concatenate((matrix,zero))
    out = matrix.reshape((length,length))
    return out


def preprocess_norm(wav):
 
    y, sr = librosa.load(wav)
    y = Normalization(y)
    processed = Normalization(y)
       
    return processed

def preprocess_fft(wav):  
    """
    processed shape: [200,1]
    """
    y, sr = librosa.load(wav)
    y = Normalization(y)
    n_fft = 2048
    ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))
    processed = ft[:200]
       
    return processed

def preprocess_stft(wav):

    y, sr = librosa.load(wav)
    y = Normalization(y)
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    processed = spec
       
    return processed

def preprocess_wavelet(wav):
    """
    suggesting padding to 150 * 150 in the dataloder
    """
    sig, fs = librosa.load(wav)
    sig = Normalization(sig)
    sig = butter_bandpass_filter(sig, 1, 3999, fs, order=3)
    wave = wavelet(sig)
    xmax=max(wave)
    xmin=min(wave)
    wave=(255-0)*(wave-xmin)/(xmax-xmin)+0       
    wave = reshape(wave)
    #display.specshow(wave)
    process = wave
    
    return process


def preprocess_mel(wav): 
    """
    
    """
    y, sr = librosa.load(wav)
    y = Normalization(y)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    processed = mel_spect
       
    return processed



if __name__ == '__main__':

    print('todo.')
