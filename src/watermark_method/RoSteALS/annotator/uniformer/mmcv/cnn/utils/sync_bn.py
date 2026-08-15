import torch

import annotator.uniformer.mmcv as mmcv

class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    
    def _check_input_dim(self, input):
        return

def revert_sync_batchnorm(module):
    
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]
    if hasattr(mmcv, 'ops'):
        module_checklist.append(mmcv.ops.SyncBatchNorm)
    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(module.num_features, module.eps,
                                     module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output
