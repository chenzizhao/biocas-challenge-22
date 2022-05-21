from base import BaseDataLoader
import torch
import torch.nn.functional as F
from torchaudio import datasets

class YesNoDataLoader(BaseDataLoader):
    """
    Our audio-equivalent of MNIST
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        def _pad_sequence(batch, ):
            # Make all tensor in a batch the same length by padding with zeros
            batch = [item.t() for item in batch]
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
            return batch.permute(0, 2, 1)

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

class RespDataLoader(BaseDataLoader):
    """
    TODO: the actual dataloader
    """
    pass
