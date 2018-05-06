import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import math


class ScaledDotProductAttn(nn.Module):

    def __init__(self, d):
        nn.Module.__init__(self)

        self.d = d
        self.scale = math.sqrt(1. / d)


    def forward(self, Q, K, V, mask=None):
        '''
        Q, K, V have the dimension of (batch_size, seq_len, dim)
        mask: (batch_size, seq_len, seq_len), mask out illegal positions
        '''
        def check(i):
            Q1, K1, V1 = Q[i:i+1], K[i:i+1], V[i:i+1]
            attn1 = torch.bmm(Q1, K1.transpose(1,2)) * self.scale
            if mask is not None:
                assert attn1.size() == mask[i:i+1].size()
                attn1.data.masked_fill_(mask[i:i+1], -float('inf'))

            attn1 = F.softmax(attn1, dim=2)
            return torch.bmm(attn1, V1)

        #outs = [check(i) for i in range(Q.size(0))]

        attn = torch.bmm(Q, K.transpose(1,2)) * self.scale
        if mask is not None:
            assert attn.size() == mask.size()
            attn.data.masked_fill_(mask, -float('inf'))

        attn = F.softmax(attn, dim=2)
        out = torch.bmm(attn, V)

        #eq = [np.asscalar((attn[i] == outs[i][0]).float().mean().data.cpu().numpy()) for i in range(Q.size(0))]
        #print('dot prod', eq)

        return out, attn


class MultiHeadAttn(nn.Module):

    def __init__(self, 
            num_head, 
            d_model, 
            d_k, 
            d_v, 
            dropout):
        '''
        num_head: int, number of heads
        d_model: int
        d_k: int, dimension of query and key
        d_v: int, dimension of value
        dropout: float, dropout rate
        '''
        nn.Module.__init__(self)
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        self.Wq = nn.Parameter(torch.FloatTensor(num_head, d_model, d_k))
        self.Wk = nn.Parameter(torch.FloatTensor(num_head, d_model, d_k))
        self.Wv = nn.Parameter(torch.FloatTensor(num_head, d_model, d_v))
        self.Wo = nn.Linear(d_v * num_head, d_model)

        self.attn = ScaledDotProductAttn(d_model)
        self.bn = nn.BatchNorm1d(d_model)

        self.__init_weights()


    def __init_weights(self):
        init.xavier_normal(self.Wq)
        init.xavier_normal(self.Wk)
        init.xavier_normal(self.Wv)
        init.xavier_normal(self.Wo.weight)


    def forward(self, Q, K, V, mask=None):
        '''
        Q, K, V: (batch_size, seq_len, dim)
        mask: (batch_size, seq_len, seq_len), mask out illegal positions
        '''
        residual = Q

        B, q_len, d = Q.size()
        k_len = K.size(1)
        v_len = V.size(1)

        def check(i):
            Q1, K1, V1 = Q[i:i+1], K[i:i+1], V[i:i+1]
            r = Q1
            Q1 = Q1.repeat(self.num_head, 1, 1).view(self.num_head, -1, d)
            K1 = K1.repeat(self.num_head, 1, 1).view(self.num_head, -1, d)
            V1 = V1.repeat(self.num_head, 1, 1).view(self.num_head, -1, d)

            Q1 = Q1.bmm(self.Wq).view(-1, q_len, self.d_k)
            K1 = K1.bmm(self.Wk).view(-1, k_len, self.d_k)
            V1 = V1.bmm(self.Wv).view(-1, v_len, self.d_v)

            out2 = self.attn(Q1, K1, V1, mask[i:i+1].repeat(self.num_head, 1, 1))
            out2 = torch.cat(torch.split(out2, 1, dim=0), dim=-1)
            out2 = self.Wo(out2)
            return out2 + r

        #outs = [check(i) for i in range(B)]

        Q = Q.repeat(self.num_head, 1, 1).view(self.num_head, -1, d)
        K = K.repeat(self.num_head, 1, 1).view(self.num_head, -1, d)
        V = V.repeat(self.num_head, 1, 1).view(self.num_head, -1, d)

        Q = Q.bmm(self.Wq).view(-1, q_len, self.d_k)
        K = K.bmm(self.Wk).view(-1, k_len, self.d_k)
        V = V.bmm(self.Wv).view(-1, v_len, self.d_v)

        if mask is not None:
            #sz = mask.size()
            #mask = mask.unsqueeze(1).repeat(1,self.num_head,1,1)
            #mask = mask.transpose(0,1).contiguous()
            #mask = mask.view(-1,sz[1],sz[2])
            mask = mask.repeat(self.num_head,1,1)

        out, attns = self.attn(Q, K, V, mask)
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)

        out = self.Wo(out)
        out = F.dropout(out, self.dropout, self.training)
        out += residual

        #eq = [np.asscalar((out[i] == outs[i][0]).float().mean().data.cpu().numpy()) for i in range(B)]
        #print('multihead attn', eq)

        if B > 1:
            out = self.bn(out.transpose(1,2).contiguous())
            out = out.transpose(1,2).contiguous()
        
        return out, attns


class PositionWiseFeedFoward(nn.Module):

    def __init__(self, input_dim, hid_dim, dropout):
        '''
        input_dim: int, dimension of input tensor
        hid_dim: int, dimension of hidden state
        dropout: float, dropout rate
        '''
        nn.Module.__init__(self)

        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, hid_dim)#nn.Conv1d(input_dim, hid_dim, 1)
        self.fc2 = nn.Linear(hid_dim, input_dim)#nn.Conv1d(hid_dim, input_dim, 1)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        '''
        x: (batch_size, seq_len, dim)
        '''
        def check(i):
            x1 = x[i:i+1]
            r = x1
            x1 = self.relu(self.fc1(x1))
            x1 = self.fc2(x1)
            return x1 + r

        #outs = [check(i) for i in range(x.size(0))]

        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, self.training)
        x += residual

        #eq = [np.asscalar((x[i] == outs[i][0]).float().mean().data.cpu().numpy()) for i in range(x.size(0))]
        #print('ff', eq)
        
        if x.size(0) > 1:
            x = self.bn(x.transpose(1,2).contiguous())
            x = x.transpose(1,2).contiguous()

        return x
