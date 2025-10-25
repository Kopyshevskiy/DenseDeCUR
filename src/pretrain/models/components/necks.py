import torch
import torch.nn as nn

def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], "Undefined init_linear: {}".format(init_linear)

    for m in module.modules(): # conv2d e mlp2 non inizializzato?? nella repo ufficiale no.
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DenseCLNeck(nn.Module):
    def __init__(self,
                 in_channels,  
                 hid_channels,
                 out_channels):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]

        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, x, avgpooled_x2]
    
def densecl_neck(in_channels, hid_channels, out_channels):
    return DenseCLNeck(in_channels, hid_channels, out_channels)