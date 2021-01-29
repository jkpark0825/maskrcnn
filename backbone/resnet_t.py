import torch
import torch.nn as nn
'''
resnet
input->out1->out2->out2+input
'''
def default_conv(in_channels, out_channels, kernel_size,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size,padding=(kernel_size//2),bias=bias,groups = groups)
def default_act():
    return nn.ReLU(True)
def default_BN(in_channels):
    return nn.BatchNorm2d(in_channels,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
def resident(in_channels,out_channels,bias=True,conv = default_conv, act = default_act, bn=default_BN,is_first=False):
    modules=[]
    if is_first is True:
        modules.append(conv(in_channels, in_channels,kernel_size=1,bias=bias))
    else:
        modules.append(conv(out_channels, in_channels,kernel_size=1,bias=bias))
    modules.append(bn(in_channels))
    modules.append(act())
    modules.append(conv(in_channels, in_channels,kernel_size=3,bias=bias))
    modules.append(bn(in_channels))
    modules.append(act())
    modules.append(conv(in_channels, out_channels,kernel_size=1,bias=bias))
    modules.append(bn(out_channels))
    if is_first is True:
        shortcut = []
        shortcut.append(conv(in_channels,out_channels,kernel_size=1))
        shortcut.append(bn(out_channels))
        shortcut = nn.Sequential(*shortcut)
    else:
        shortcut = None
    body = nn.Sequential(*modules)
    return body, shortcut

def deconv(n_feats):
    deconv_kargs = {'stride': 2, 'padding': 1, 'output_padding': 1}
    return nn.Sequential(
        nn.ConvTranspose2d(n_feats, int(n_feats/2), 3, **deconv_kargs),
        nn.ConvTranspose2d(int(n_feats/2), int(n_feats/4), 3, **deconv_kargs),
        nn.Conv2d(int(n_feats/4), 3, 5, stride=1, padding=2),
    )

class resblock(nn.Module):
    def __init__(self,in_channels,out_channels,bias=True,conv = default_conv, act = default_act, bn=default_BN):
        super(resblock,self).__init__()
        self.body1, self.shortcut1 =  resident(in_channels,out_channels,is_first=True)
        self.body2, _ =  resident(in_channels,out_channels)
        self.body3, _ =  resident(in_channels,out_channels)
        
        self.act = act()
        
    def forward(self, input):
        res = self.body1(input)
        res = self.shortcut1(input)+res
        res = self.act(res)
        shortcut = res
        
        res = self.body2(res)
        res = shortcut+res
        res = self.act(res)
        shortcut = res

        res = self.body3(res)
        res = shortcut+res
        res = self.act(res)
        shortcut = res

        return res

        