import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class DRIVEDataset(CustomDataset):
    
    CLASSES = ('background', 'vessel')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(DRIVEDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_manual1.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
