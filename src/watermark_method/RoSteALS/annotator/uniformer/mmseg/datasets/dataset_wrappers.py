from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS

@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE

@DATASETS.register_module()
class RepeatDataset(object):
    
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        
        return self.times * self._ori_len
