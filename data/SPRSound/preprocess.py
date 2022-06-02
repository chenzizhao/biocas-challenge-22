from os import listdir
from os.path import join
from torchaudio import load
import numpy as np

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

def preprocess_norm(wav):
    """
    This function will be exported to main.py
    """
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
    """
    This function will be exported to main.py
    """
    y, sr = librosa.load(wav)
    y = Normalization(y)
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    processed = spec
       
    return processed


def preprocess_mel(wav):
    """
    This function will be exported to main.py
    """
    y, sr = librosa.load(wav)
    y = Normalization(y)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    processed = mel_spect
       
    return processed

if __name__ == '__main__':

    print('todo.')
