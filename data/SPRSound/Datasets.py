from os.path import join
from torch.utils.data import Dataset
import pandas as pd
import torch

class Resp21Dataset(Dataset):
    def __init__(self, data_dir):
        self.csv = pd.read_csv(join(data_dir, 'task2.csv'))
        self.audio_dir = join(data_dir, 'processed')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        entry = self.csv.iloc[index]
        wav_name, target = entry['wav_name'], entry['label_21']
        wav = torch.load(join(self.audio_dir, wav_name))
        return wav, target

if __name__ == "__main__":
    r21 = Resp21Dataset('.')
    print(f"There are {len(r21)} samples in the dataset.")
    wav, target = r21[0]
    print(target)
    print(wav.shape)