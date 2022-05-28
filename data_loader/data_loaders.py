from ntpath import join
from os import listdir
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from data.SPRSound.preprocess import preprocess
from torchaudio import datasets, load
import importlib

class YesNoDataLoader(BaseDataLoader):
    """
    Our audio-equivalent of MNIST
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        def collate_fn(batch):
            tensors, targets = [], []

            # Gather in lists, and encode labels as indices
            for waveform, sample_rate, label in batch:
                tensors += [waveform]
                targets += [torch.Tensor(label)]

            # Group the list of tensors into a batched tensor
            #   hack to ensure always 55840
            tensors[0] = F.pad(tensors[0], (0, 55840-tensors[0].shape[-1]), mode='constant', value=0.)
            tensors = _pad_sequence(tensors)
            tensors.squeeze_(1) # 54x55840
            
            targets = torch.stack(targets) # 54x8

            return tensors, targets

        self.data_dir = data_dir
        self.dataset = datasets.YESNO(self.data_dir, download=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

class Resp21DataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.CLASSES = ('Normal', 'Adventitious', 'Poor Quality')
        self.CLASS2INT = {'Normal':0, 'Adventitious':1, 'Poor Quality':2}

        def collate_fn(batch):
            tensors, targets = [], []

            # Gather in lists, and encode labels as indices
            for wave, label in batch:
                tensors += [wave]
                targets += [torch.LongTensor([self.CLASS2INT[label]])]
            # Group the list of tensors into a batched tensor
            # -- again hacking here, we might want variable length
            tensors[0] = F.pad(tensors[0], (0, 122880-tensors[0].shape[-1]), mode='constant', value=0.)
            
            tensors = _pad_sequence(tensors)
            tensors.squeeze_(1)
            targets = torch.stack(targets)
            targets.squeeze_(1)
            return tensors, targets

        self.data_dir = data_dir
        Datasets = _import_dataset_module(data_dir)
        self.dataset = Datasets.Resp21Dataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

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
                # return preprocess(wav)

                # hack
                wav = F.pad(wav, (0, 122880-wav.shape[-1]), mode='constant', value=0.)
                return fname, wav

        self.data_dir = data_dir
        self.dataset = MainDataSet()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

def _pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def _import_dataset_module(data_dir):
    # Dynamically load `Datasets.py` from `data_dir`
    dataset_dir = join(data_dir, 'Datasets.py')
    spec=importlib.util.spec_from_file_location("Datasets",dataset_dir)
    Datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Datasets)
    return Datasets
