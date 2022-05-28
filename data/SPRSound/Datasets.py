from os.path import join
from torch.utils.data import Dataset
from torchaudio import load
import pandas as pd


class Resp21Dataset(Dataset):
    def __init__(self, data_dir):
        self.csv = pd.read_csv(join(data_dir, 'task2.csv'))
        self.audio_dir = join(data_dir, 'wav')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        entry = self.csv.iloc[index]
        wav_name = entry['wav_name']
        audio_sample_path = join(self.audio_dir, wav_name)
        wav, sample_rate = load(audio_sample_path)
        target = entry['label_21']
        return wav, target

if __name__ == "__main__":
    r21 = Resp21Dataset('.')
    print(f"There are {len(r21)} samples in the dataset.")
    wav, _, target = r21[0]
    print(target)
    print(wav.shape)  # max length 122880
    