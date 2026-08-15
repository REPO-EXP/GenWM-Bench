import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

EPS = 1e-10

class VggFeatureExtractor(nn.Module):
    def __init__(self):
        super(VggFeatureExtractor, self).__init__()
        
        vgg16 = torchvision.models.vgg16(pretrained=True).features
        
        for param in vgg16.parameters():
            param.requires_grad = False
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(4): 
            self.slice1.add_module(str(x), vgg16[x])
        for x in range(4, 9): 
            self.slice2.add_module(str(x), vgg16[x])
        for x in range(9, 16): 
            self.slice3.add_module(str(x), vgg16[x])
        for x in range(16, 23): 
            self.slice4.add_module(str(x), vgg16[x])
        for x in range(23, 30): 
            self.slice5.add_module(str(x), vgg16[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

def normalize_tensor(t):
    
    N, C, H, W = t.shape
    norm_factor = torch.sqrt(torch.sum(t**2,dim=1)).view(N,1,H,W)
    return t/(norm_factor.expand_as(t)+EPS)

def softmax(a, b, factor=1):
    concat = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
    softmax_factors = F.softmax(concat * factor, dim=-1)
    return a * softmax_factors[:,:,:,:,0] + b * softmax_factors[:,:,:,:,1]

class WatsonDistanceVgg(nn.Module):
    
    def __init__(self, trainable=False, reduction='sum'):
        
        super().__init__()
        
        self.add_module('vgg', VggFeatureExtractor())
        
        self.shift = nn.Parameter(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False)
        self.scale = nn.Parameter(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False)
            
        self.L = 5
        self.channels = [64,128,256,512,512]
        
        self.t0_tild = nn.Parameter(torch.zeros((self.channels[0])), requires_grad=trainable)
        self.t1_tild = nn.Parameter(torch.zeros((self.channels[1])), requires_grad=trainable)
        self.t2_tild = nn.Parameter(torch.zeros((self.channels[2])), requires_grad=trainable)
        self.t3_tild = nn.Parameter(torch.zeros((self.channels[3])), requires_grad=trainable)
        self.t4_tild = nn.Parameter(torch.zeros((self.channels[4])), requires_grad=trainable)
            
        w = torch.tensor(0.2) 
        self.w0_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable) 
        self.w1_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable)
        self.w2_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable)
        self.w3_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable)
        self.w4_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable)
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=trainable) 
        
        self.dropout = nn.Dropout(0.5 if trainable else 0)
        
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception('Reduction "{}" not supported. Valid values are: "sum", "none".'.format(reduction))

    @property
    def t(self):
        return [torch.exp(t) for t in [self.t0_tild, self.t1_tild, self.t2_tild, self.t3_tild, self.t4_tild]]
    
    @property
    def w(self):
        
        return [torch.sigmoid(w) for w in [self.w0_tild, self.w1_tild, self.w2_tild, self.w3_tild, self.w4_tild]]
    
    def forward(self, input, target):
        
        input = (input - self.shift.expand_as(input))/self.scale.expand_as(input)
        target = (target - self.shift.expand_as(target))/self.scale.expand_as(target)
        
        c0 = self.vgg(target)
        c1 = self.vgg(input)
        
        for l in range(self.L):
            c0[l] = normalize_tensor(c0[l])
            c1[l] = normalize_tensor(c1[l])
        
        t = self.t
        w = self.w
        s = []
        for l in range(self.L):
            N, C_l, H_l, W_l = c0[l].shape
            t_l = t[l].view(1,C_l,1,1).expand(N, C_l, H_l, W_l)
            s.append(softmax(t_l, (c0[l].abs() + EPS)**w[l] * t_l**(1 - w[l])))
        
        watson_dist = 0
        for l in range(self.L):
            _, _, H_l, W_l = c0[l].shape
            layer_dist = (((c0[l] - c1[l]) / s[l]).abs() + EPS) ** self.beta
            layer_dist = self.dropout(layer_dist) + EPS
            layer_dist = torch.sum(layer_dist, dim=(1,2,3)) 
            layer_dist = (1 / (H_l * W_l)) * layer_dist 
            watson_dist += layer_dist  
        watson_dist = watson_dist ** (1 / self.beta)

        if self.reduction == 'sum':
            watson_dist = torch.sum(watson_dist)
        
        return watson_dist
