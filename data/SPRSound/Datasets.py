from os.path import join
from torch.utils.data import Dataset
import pandas as pd
import torch

class RespDataset(Dataset):
    def __init__(self, data_dir, task=1):
        assert task in (1,2)
        self.task = task
        self.csv = pd.read_csv(join(data_dir, f'task{task}.csv'))
        self.dir = join(data_dir, 'processed_ast')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        entry = self.csv.iloc[index]
        wav_name = entry['wav_name']
        target = (entry[f'label_{self.task}1'], entry[f'label_{self.task}2'])
        wav = torch.load(join(self.dir, wav_name))
        ##normolize
        #wav = (wav-37.3)/(2.3*2)
        wav = wav.to(torch.float32)
        return wav, target

        return wav, target

if __name__ == "__main__":
    r = RespDataset('.', task=1)
    print(f"There are {len(r)} samples in the dataset.")
    wav, target = r[0]
    print(target)
    print(wav.shape)
