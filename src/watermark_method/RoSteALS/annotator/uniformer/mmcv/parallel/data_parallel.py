
from itertools import chain

from torch.nn.parallel import DataParallel

from .scatter_gather import scatter_kwargs

class MMDataParallel(DataParallel):
    
    def __init__(self, *args, dim=0, **kwargs):
        super(MMDataParallel, self).__init__(*args, dim=dim, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        
        if not self.device_ids:
            
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
        else:
            return super().forward(*inputs, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.train_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1,            ('MMDataParallel only supports single GPU training, if you need to'
             
             )

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    
                    )

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        if not self.device_ids:
            
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.val_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1,            ('MMDataParallel only supports single GPU training, if you need to'
             
             )

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    
                    )

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])
