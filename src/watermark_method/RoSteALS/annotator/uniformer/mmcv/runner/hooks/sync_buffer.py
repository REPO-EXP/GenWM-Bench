
from ..dist_utils import allreduce_params
from .hook import HOOKS, Hook

@HOOKS.register_module()
class SyncBuffersHook(Hook):
    
    def __init__(self, distributed=True):
        self.distributed = distributed

    def after_epoch(self, runner):
        
        if self.distributed:
            allreduce_params(runner.model.buffers())
