'''
Three functions (save_prc_stft,save_prc_wavelet,save_prc_mel) that transform the wave to the image in the specific folder
One function preprocess_img that preprosses the images above before read by dataloader 
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import pywt
import math
import os
import librosa.display as display
from scipy.signal import butter, lfilter
import cv2

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

def Normalization(x):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    for i in range(len(x)):
        
        x[i] = float(x[i]-min_x)/(max_x-min_x)
           
    return x


def save_pic_wavelet(wav_dir,save_dir):

    for file in os.listdir(wav_dir):          
        fs,sig= wav.read(wav_dir+'/'+file)
        sig = Normalization(sig)
        if fs>4000:
            sig = butter_bandpass_filter(sig, 1, 3999, fs, order=3)
            
        wave = wavelet(sig)
        xmax=max(wave)
        xmin=min(wave)
        wave=(255-0)*(wave-xmin)/(xmax-xmin)+0       
        wave = reshape(wave)
        display.specshow(wave)
        plt.rcParams['figure.dpi'] = 100  
        plt.rcParams['figure.figsize'] = (2.24, 2.24)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        
        plt.savefig(save_dir+'/'+file[:-3]+'png', cmap='Greys_r')
        plt.close()
  
def save_pic_stft(wav_dir,save_dir):

    for file in os.listdir(wav_dir):           
        fs,sig= wav.read(wav_dir+'/'+file)
        sig = Normalization(sig)
        # if fs>8000:
        sig = butter_bandpass_filter(sig, 1, 3999, fs, order=3)
        stft = librosa.stft(sig, n_fft=int(0.02*fs), hop_length=int(0.01*fs), window='hann')
        # if fs>8000:
        display.specshow(librosa.amplitude_to_db(stft[0:int(len(stft)/2),:],ref=np.max),y_axis='log',x_axis='time')
        # else:
        #display.specshow(librosa.amplitude_to_db(stft,ref=np.max),y_axis='log',x_axis='time')
        plt.rcParams['figure.dpi'] = 100  
        plt.rcParams['figure.figsize'] = (2.24, 2.24)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.savefig(save_dir+'/'+file[:-3]+'png', cmap='Greys_r')
        plt.close()
        
 def save_pic_mel(wav_dir,save_dir):
    
    for file in os.listdir(wav_dir):           
        sr,y = wav.read(wav_dir+'/'+file)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        log_melspec = librosa.amplitude_to_db(mel_spect)
        # librosa.display.specshow(librosa.amplitude_to_db(mel_spect[0:int(len(mel_spect)/2),:],ref=np.max),y_axis='log',x_axis='time')
        librosa.display.specshow(mel_spect[:50], y_axis='mel', fmax=8000, x_axis='time')
        plt.rcParams['figure.dpi'] = 100  
        plt.rcParams['figure.figsize'] = (2.24, 2.24)
        plt.margins(0,0)
        plt.axis('off')
        plt.savefig(save_dir+'/'+file[:-3]+'png', cmap='Greys_r')
        plt.close()




if __name__ == '__main__':
    save_pic_wavelet('','')
    save_pic_stft('','')
    save_pic_mel('','')
