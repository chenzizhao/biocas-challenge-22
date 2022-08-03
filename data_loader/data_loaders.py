from base import BaseDataLoader
import torch
from data.SPRSound import Datasets
import torch.nn.functional as F


# Important Assumption (used in model/metric.py)
# Normal is always index 0
# PQ, if exists, is index 1

def resp_classes(task, level):
    assert task in (1,2), 'Task has to be either 1 or 2.'
    assert level in (1,2), 'Level has to be either 1 or 2.'
    if task==1:
        if level==1:
            CLASSES = ('Normal', 'Adventitious')
        elif level==2:
            CLASSES = ('Normal', 'Rhonchi', 'Wheeze', 'Stridor', 'Coarse Crackle', 'Fine Crackle', 'Wheeze & Crackle')
    elif task==2:
        if level==1:
            CLASSES = ('Normal', 'Poor Quality', 'Adventitious')
        elif level==2:
            CLASSES = ('Normal', 'Poor Quality', 'CAS', 'DAS', 'CAS & DAS')
    return CLASSES

class RespDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, task=1, level=1, input_dir='processed'):
        self.CLASSES = resp_classes(task, level)
        self.CLASS2INT = {label:i for (i, label) in enumerate(self.CLASSES)}
        self.LEVEL = level

        dataset = Datasets.RespDataset(data_dir, task=task, input_dir=input_dir)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for wave, label in batch:
            label = label[self.LEVEL-1]
            tensors += [wave]
            targets += [torch.LongTensor([self.CLASS2INT[label]])]
        # Group the list of tensors into a batched tensor
        tensors = torch.stack(tensors)
        targets = torch.stack(targets)
        targets.squeeze_(1)
        return tensors, targets

class PadDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, task=1, level=1, padding_max=122880):
        self.CLASSES = resp_classes(task, level)
        self.CLASS2INT = {label:i for (i, label) in enumerate(self.CLASSES)}
        self.LEVEL = level
        self.padding_max = padding_max

        dataset = Datasets.RespDataset(data_dir, task=task)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for wave, label in batch:
            label = label[self.LEVEL-1]
            tensors += [wave]
            targets += [torch.LongTensor([self.CLASS2INT[label]])]
        # Group the list of tensors into a batched tensor
        # -- again hacking here, we might want variable length
        tensors[0] = F.pad(tensors[0], (0, self.padding_max-tensors[0].shape[-1]), mode='constant', value=0.)

        tensors = _pad_sequence(tensors)
        tensors.squeeze_(1)
        targets = torch.stack(targets)
        targets.squeeze_(1)
        return tensors, targets

def _pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)
