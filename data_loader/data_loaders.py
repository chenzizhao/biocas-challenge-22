from ntpath import join
from os import listdir
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from data.SPRSound.preprocess import preprocess
from torchaudio import datasets, load
import importlib


# Important Assumption (used in model/metric.py)
# Normal is always index 0
# PQ, if exists, is index 1

class Resp21DataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.CLASSES = ('Normal', 'Poor Quality', 'Adventitious')
        self.CLASS2INT = {label:i for (i, label) in enumerate(self.CLASSES)}

        def collate_fn(batch):
            tensors, targets = [], []

            # Gather in lists, and encode labels as indices
            for wave, label in batch:
                tensors += [wave]
                targets += [torch.LongTensor([self.CLASS2INT[label]])]
            # Group the list of tensors into a batched tensor
            tensors = torch.stack(tensors)
            tensors.squeeze_(1)
            targets = torch.stack(targets)
            targets.squeeze_(1)
            return tensors, targets

        Datasets = _import_dataset_module(data_dir)
        dataset = Datasets.Resp21Dataset(data_dir)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

class MainDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size=1, shuffle=False, validation_split=0.0, num_workers=1, training=False):
        self.CLASSES = ('Normal', 'Adventitious', 'Poor Quality')

        class MainDataSet(Dataset):
            def __init__(self):
                self.audio_dir = data_dir
                self.fnames = listdir(self.audio_dir)

            def __len__(self):
                return len(self.fnames)

            def __getitem__(self, index):
                fname = self.fnames[index]
                wav_path = join(self.audio_dir, fname)
                wav, sample_rate = load(wav_path)
                return fname, preprocess(wav)

        self.data_dir = data_dir
        self.dataset = MainDataSet()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

def _import_dataset_module(data_dir):
    # Dynamically load `Datasets.py` from `data_dir`
    dataset_dir = join(data_dir, 'Datasets.py')
    spec=importlib.util.spec_from_file_location("Datasets",dataset_dir)
    Datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Datasets)
    return Datasets
