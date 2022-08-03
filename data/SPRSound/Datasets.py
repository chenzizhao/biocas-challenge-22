from os.path import join
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchaudio

class RespDataset(Dataset):
    def __init__(self, data_dir, task, input_dir=None):
        assert task in (1,2)
        self.task = task
        self.csv = pd.read_csv(join(data_dir, f'task{task}.csv'))
        self.input_dir = input_dir
        if input_dir is None:
            if task == 1:
                self.dir = join(data_dir, 'clip')
            else:
                self.dir = join(data_dir, 'wav')
        else:
            self.dir = join(data_dir, input_dir)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        entry = self.csv.iloc[index]
        wav_name = entry['wav_name']
        target = (entry[f'label_{self.task}1'], entry[f'label_{self.task}2'])
        if self.input_dir is None:
            wav, _ = torchaudio.load(join(self.dir, wav_name))
        else:
            wav = torch.load(join(self.dir, wav_name)).to(torch.float32)
            ##normolize
            #wav = (wav-37.3)/(2.3*2)
        return wav, target

if __name__ == "__main__":
    r = RespDataset('.', task=1, raw=True)
    print(f"There are {len(r)} samples in the dataset.")
    wav, target = r[0]
    print(target)
    print(wav.shape)
