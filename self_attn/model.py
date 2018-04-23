import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

    def __init__(self, 
            num_classes, 
            num_channels, 
            feature_maps, 
            num_heads, 
            num_layers, 
            d_k, 
            d_v, 
            hid_dim, 
            fc_hids, 
            dropout
            ):
        nn.Module.__init__(self)

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # convolution modules
        self.conv1 = nn.Conv2d(num_channels, feature_maps[0], 5)
        self.conv2 = nn.Conv2d(feature_maps[0], feature_maps[1], 5)

        # self-attention modules
        dummy = Variable(torch.zeros(1, num_channels, 128, 128))
        dummy = self.patched_conv(dummy)
        _, num_patches, input_dim = dummy.size()

        self.attn = nn.ModuleList([
            MultiHeadAttn(num_heads, input_dim, d_k, d_v, dropout)
            for i in range(num_layers)])
        self.ff = nn.ModuleList([
            PositionWiseFeedFoward(input_dim, hid_dim, dropout) 
            for i in range(num_layers)])

        # fully-connected layer to classify
        fc_hids = [input_dim * num_patches] + fc_hids + [num_classes]
        self.fc = nn.ModuleList([
                nn.Linear(fc_hids[i], fc_hids[i+1]) 
                for i in range(len(fc_hids)-1)
                ])


    def forward(self, x):
        '''
        x: (B, C, H, W) where H = W = 128
        '''
        assert x.size(2) == x.size(3) == 128, \
                '[DataError] Image size must be 128 x 128!'

        x = self.patched_conv(x)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_layers))

        attns = []
        for attn, ff in zip(self.attn, self.ff):
            x, attn_i = attn(x, x, x)
            attns.append(attn_i)
            x = ff(x)

        x = x.view(x.size(0), -1)
        for i in range(len(self.fc)-1):
            x = self.fc[i](x)
            x = F.dropout(x, self.dropout, self.training)
        x = self.fc[-1](x)

        return x, attns


    def patchify(self, x):
        '''
        split image x to 4 x 4 patches, each patch is 32 x 32.
        '''
        B, C, H, W = x.size()
        h, w = 32, 32
        patches = []
        for i in range(H//h):
            x_row_i = x[:,:,i*h:(i+1)*h,:]
            patches.extend(x_row_i.split(w, -1))
        return patches


    def CONV(self, x):
        '''
        Simple two-layer convolution with kernel size 5, max pooling and relu
        '''
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = GradMultiply.apply(x, 1.0 / 16)
        return x


    def patched_conv(self, patches):
        patches = self.patchify(patches)
        patches = [self.CONV(p) for p in patches]
        patches = torch.stack(patches, 1)
        return patches

