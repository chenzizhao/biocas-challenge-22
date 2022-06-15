from os.path import join
from base import BaseDataLoader
import torch
import importlib


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

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, task=1, level=1):
        self.CLASSES = resp_classes(task, level)
        self.CLASS2INT = {label:i for (i, label) in enumerate(self.CLASSES)}

        def collate_fn(batch):
            tensors, targets = [], []

            # Gather in lists, and encode labels as indices
            for wave, label in batch:
                label = label[level-1]
                tensors += [wave]
                targets += [torch.LongTensor([self.CLASS2INT[label]])]
            # Group the list of tensors into a batched tensor
            tensors = torch.stack(tensors)
            targets = torch.stack(targets)
            targets.squeeze_(1)
            return tensors, targets

        Datasets = _import_dataset_module(data_dir)
        dataset = Datasets.RespDataset(data_dir, task=task)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

def _import_dataset_module(data_dir):
    # Dynamically load `Datasets.py` from `data_dir`
    dataset_dir = join(data_dir, 'Datasets.py')
    spec=importlib.util.spec_from_file_location("Datasets",dataset_dir)
    Datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Datasets)
    return Datasets
