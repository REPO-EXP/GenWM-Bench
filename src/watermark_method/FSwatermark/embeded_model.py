import torch
import torch.nn as nn

def clamp_x(data):
    return (data-torch.min(data))/(torch.max(data)-torch.min(data))

class SEncoder(nn.Module):
    def __init__(self, image_size, secret_size):
        super(SEncoder, self).__init__()
        self.image_size = image_size
        self.secret_size = secret_size
        rate = 2
        self.linear_1 = nn.Sequential(nn.Linear(in_features=secret_size, out_features=int(1024*rate)),
                                      nn.SELU(inplace=True),
                                      nn.Linear(in_features=int(1024*rate), out_features=int(1280*rate)),
                                      nn.SELU(inplace=True))
        self.linear_3 = nn.Sequential(nn.Linear(in_features=int(1280*rate), out_features=int(1024*rate)),
                                      nn.SELU(inplace=True),
                                      nn.Linear(in_features=int(1024*rate), out_features=int(1024*rate)),
                                      nn.SELU(inplace=True),
                                      nn.Linear(in_features=int(1024*rate), out_features=64*64))

        self.pre_conv = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                                padding=1),
                                      nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True),
                                      nn.SELU(inplace=True),
                                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                                padding=1),
                                      nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True),
                                      nn.SELU(inplace=True),
                                      nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                                padding=1),
                                      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True),
                                      nn.SELU(inplace=True),
                                      nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                                padding=1),
                                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True),
                                      nn.SELU(inplace=True),
                                      nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                                padding=1),
                                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True),
                                      nn.SELU(inplace=True),
                                      nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(3, 3), stride=(1, 1),
                                                padding=1),
                                      nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True),)

    def forward(self, s, mask_all=None):
        s_t = self.linear_1(s)
        secret_2 = s_t * 1
        s_v = self.linear_3(secret_2)
        
        txt_secret_embedding = s_t.squeeze(1)
        s_v = torch.reshape(s_v, (s_v.shape[0], 1, 64, 64))
        s_v = s_v.repeat(1, 4, 1, 1)
        s_v = self.pre_conv(s_v)

        return txt_secret_embedding, s_v

class SDecoder(nn.Module):
    def __init__(self, image_size, secret_size):
        super(SDecoder, self).__init__()
        self.image_size = image_size
        self.secret_size = secret_size
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1),
                                  nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True,
                                                 track_running_stats=True),
                                  nn.SELU(inplace=True),
                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1),
                                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                 track_running_stats=True),
                                  nn.SELU(inplace=True),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1),
                                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                 track_running_stats=True),
                                  nn.SELU(inplace=True),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1),
                                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                 track_running_stats=True),
                                  nn.SELU(inplace=True),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1),
                                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                 track_running_stats=True),
                                  nn.SELU(inplace=True),
                                  nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1),
                                  nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True,
                                                 track_running_stats=True),
                                  nn.SELU(inplace=True),
                                  nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(2, 2),
                                            padding=1)
                                  )
        self.ave_pool2d = nn.AdaptiveAvgPool2d(output_size=(256, 256))
        self.linear_1 = nn.Sequential(nn.Linear(in_features=int(self.image_size/2)**2, out_features=1024),
                                      nn.SELU(),
                                      nn.Linear(in_features=1024, out_features=512),
                                      nn.SELU(inplace=True),
                                      nn.Linear(in_features=512, out_features=self.secret_size))

    def forward(self, images):
        x = self.conv(images)
        x = self.ave_pool2d(x)
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
        out_put = torch.sigmoid(self.linear_1(x)*10)
        return out_put
