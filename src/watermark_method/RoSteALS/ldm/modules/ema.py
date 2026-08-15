import torch
import torch.nn as nn

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        self.decay = decay
        self.use_num_upates = use_num_upates
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}
        self.num_updates = nn.Parameter(torch.zeros(1), requires_grad=False)
    def store(self, parameters):
        self.shadow = {k: v.clone().detach() for k, v in parameters}
    def restore(self, parameters):
        for name, param in parameters:
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
    def __call__(self, model):
        if self.use_num_upates: self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates)) if self.use_num_upates else self.decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].data.copy_(d * self.shadow[name] + (1 - d) * param.data)
                    param.data.copy_(self.shadow[name])
