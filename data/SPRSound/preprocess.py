"""
The `preprocess` function will be exported to main.py

Recording (Task 2-1 and Task 2-2):
- preprocee_fft,
- preprocess_norm,
- maybe (preprocee_stft, preprocess_mel, preprocess_wavelet)

Event (Task 1-1 and Task 1-2):
- preprocee_stft
- preprocess_mel
- preprocess_wavelet

"""

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchaudio import load
import pywt
import librosa
import librosa.display as display
from scipy.signal import butter, lfilter
from os import listdir, makedirs
from os.path import join, exists
import tempfile
from tqdm import tqdm

def normalize(x):
    maximum = np.max(x)
    minimum = np.min(x)
    return (x-minimum) / (maximum - minimum)

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

def _display_to_tensor_routine(size=(224, 224)):
    """
    This is a subroutine to convert plt figures to an image tensor, with size torch.Size([3, 224, 224])
    """
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)

    # Hack to convert plt.figure to an image tensor
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'w') as f:
        plt.savefig(tmp.name + '.png')
        plt.close()
        img = plt.imread(tmp.name + '.png')
    
    # torch.Size([H, W, C])
    img_tensor = torch.from_numpy(np.array(img))
    # convert RGBA to RGB by dropping the last channel
    img_tensor = img_tensor[:,:,[0,1,2]]
    # torch.Size([C, H, W])
    img_tensor = torch.permute(img_tensor, (2, 0, 1))
    # interpolate to get the desired size
    img_tensor.unsqueeze_(0)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=size)
    img_tensor.squeeze_(0)
    
    # # To view the image itself, try uncomment the following:
    # plt.imshow(torch.permute(img_tensor, (1, 2, 0)))
    # plt.show()

    # assert img_tensor.shape == (3, size[0], size[1])

    # Float between [0, 1]
    return img_tensor


def preprocess_stft(wav, sample_freq=8000):
    sig = wav
    fs = sample_freq
    sig = normalize(sig)
    sig = butter_bandpass_filter(sig, 1, 3999, fs, order=3)
    wav = librosa.stft(sig, n_fft=int(0.02*fs), hop_length=int(0.01*fs), window='hann')
    wav = librosa.amplitude_to_db(wav[0:int(len(wav)/2),:],ref=np.max)
    display.specshow(wav, y_axis='log',x_axis='time')
    img_tensor = _display_to_tensor_routine()
    return img_tensor

def preprocess_wavelet(wav, sample_freq=8000):
    sig = wav
    fs = sample_freq
    sig = normalize(sig)
    sig = butter_bandpass_filter(sig, 1, 3999, fs, order=3)
    wave = wavelet(sig)
    xmax = max(wave)
    xmin = min(wave)
    wave = (255-0)*(wave-xmin)/(xmax-xmin)+0
    wave = reshape(wave)
    display.specshow(wave)
    img_tensor = _display_to_tensor_routine()
    return img_tensor

def preprocess(wav):
    """
    Input: wav as a np.ndarray
    Output: single tensor as input of classifier.
    --------------------
    This is a simple wrap function to provide a unifying API
    """
    return preprocess_stft(wav, sample_freq=8000)

if __name__ == '__main__':
    REC_DIR = "wav"
    CLIP_DIR = "clip"
    PROC_DIR = "processed"

    if not exists(PROC_DIR):
        makedirs(PROC_DIR)

    for dir in (REC_DIR, CLIP_DIR):
        print(f"Processing waves in {dir}/ folder")
        for wav_name in tqdm(listdir(dir)):
            wav, _ = load(join(dir, wav_name))
            wav = wav.squeeze().cpu().detach().numpy()
            processed = preprocess(wav)
            torch.save(processed, join(PROC_DIR, wav_name))
