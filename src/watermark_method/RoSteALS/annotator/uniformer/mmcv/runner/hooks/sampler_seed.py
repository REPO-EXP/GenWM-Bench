
from .hook import HOOKS, Hook

@HOOKS.register_module()
class DistSamplerSeedHook(Hook):
    
    def before_epoch(self, runner):
        if hasattr(runner.data_loader.sampler, 'set_epoch'):
            
            runner.data_loader.sampler.set_epoch(runner.epoch)
        elif hasattr(runner.data_loader.batch_sampler.sampler, 'set_epoch'):
            
            runner.data_loader.batch_sampler.sampler.set_epoch(runner.epoch)
