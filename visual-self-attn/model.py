import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from modules import *


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class VisualSelfAttn(nn.Module):
    '''
    Args
        - num_classes: int, number of classes
        - num_heads:   int, number of multi-head in self-attention
        - num_layers:  int, number of self-attention layers
        - d_k, d_v:    int, dimension of key and value, typically have the same 
                            value, and d_k(d_v) x num_heads = 512
        - hid_dim:     int, hidden dimension in position-wise feed-forward net
        - dropout:     float, range in [0, 1], dropout rate
        - mode:        str, valid for 'vgg', 'mean' or 'flatten'
                            * vgg: same as original VGG net, except the last
                                   fully-connected layer
                            * mean: uses self-attention after vgg16.features, 
                                    takes average along the length dimension 
                                    before fully-connected layers
                            * flatten: uses self-attention after vgg16.features, 
                                    flatten the extracted features before 
                                    fully-connected layers
    '''

    def __init__(self, 
            num_classes, 
            num_heads, 
            num_layers, 
            d_k, 
            d_v, 
            hid_dim, 
            dropout, 
            mode='mean'
            ):
        nn.Module.__init__(self)
        
        # VGG pretrain
        vgg16 = models.vgg16(pretrained=True)
        self.encoder = vgg16.features
        input_dim = 512

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode

        # self-attention modules
        if model != 'vgg':
            self.attn = nn.ModuleList([
                MultiHeadAttn(num_heads, input_dim, d_k, d_v, dropout)
                for i in range(num_layers)])
            self.ff = nn.ModuleList([
                PositionWiseFeedFoward(input_dim, hid_dim, dropout) 
                for i in range(num_layers)])

        # fully-connected layer to classify
        if mode == 'mean':
            hid = 128
            self.fc = nn.Sequential(
                    nn.Linear(input_dim, hid), 
                    nn.ReLU(), 
                    nn.Dropout(dropout), 
                    nn.Linear(hid, num_classes)
                    )
        else:
            fcs = [fc for fc in vgg16.classifier][:-1]
            self.fc = nn.Sequential(
                    *fcs, 
                    nn.Linear(4096, 512), 
                    nn.ReLU(), 
                    nn.Dropout(dropout), 
                    nn.Linear(512, num_classes)
                    )


    def forward(self, x):
        x = self.encoder(x)

        attns = []
        if self.mode != 'vgg':
            x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
            for attn, ff in zip(self.attn, self.ff):
                x, attn_i = attn(x, x, x)
                attns.append(attn_i)
                x = ff(x)

        if self.mode == 'mean':
            x = x.mean(1)
        elif self.mode == 'flatten':
            x = x.transpose(1, 2).contiguous()
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, attns
