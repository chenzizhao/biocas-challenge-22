from os import listdir
from os.path import join
from base import BaseDataLoader
import torch
import torch.nn.functional as F
from torchvision import datasets as datasetsV, transforms
from torchaudio import datasets as datasetsA, load

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasetsV.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


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
        self.dataset = datasetsA.YESNO(self.data_dir, download=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

class MainDataLoader(BaseDataLoader):
    """
    The dataloader for main.py
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=False):
        
        tensors = []
        # Gather in lists, and encode labels as indices
        for fname in listdir(data_dir):
            f = join(data_dir, fname)
            waveform, sample_rate = load(f)
            tensors += [waveform]

        # Group the list of tensors into a batched tensor
        tensors = _pad_sequence(tensors)

        super().__init__(tensors, batch_size, shuffle, validation_split, num_workers)

class RespDataLoader(BaseDataLoader):
    """
    TODO: the actual dataloader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        pass

def _pad_sequence(batch, ):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)
