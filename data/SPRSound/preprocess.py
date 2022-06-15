"""
The `preprocess` function will be exported to main.py
"""

from os import listdir
from os.path import join
from torchaudio import load
import torch
import librosa
from tqdm import tqdm

def normalize(x):
    maximum = torch.max(x, dim=-1, keepdim=True)[0]
    minimum = torch.min(x, dim=-1, keepdim=True)[0]
    return (x-minimum) / (maximum - minimum)

def preprocess(wav):
    """
    Input: torch.Tensor of torch.Size([1, N]), where N is total number frames
    Output: torch.Tensor of size torch.Size([200])
    """
    wav = normalize(wav)
    wav = torch.squeeze(wav)
    # librosa uses np.ndarry exclusively
    wav = wav.cpu().detach().numpy()
    n_fft = 2048
    ft = librosa.stft(wav[:n_fft], hop_length=n_fft+1)
    ft = ft[:200]
    # convert back to torch.Tensor
    processed = torch.from_numpy(ft)
    processed = torch.abs(processed)
    processed = torch.squeeze(processed)
    return processed

if __name__ == '__main__':
    WAV_DIR = "wav"
    CLIP_DIR = "clip"
    PROC_DIR = "processed"

    for wav_name in tqdm(listdir(WAV_DIR)):
        wav, _ = load(join(WAV_DIR, wav_name))
        processed = preprocess(wav)
        torch.save(processed, join(PROC_DIR, wav_name))
        # print(processed.shape)
        # break
