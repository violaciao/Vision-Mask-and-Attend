import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


class Decoder(nn.Module):

    def __init__(self, 
            vocab_size, 
            emb_dim, 
            enc_dim, 
            hid_dim, 
            dropout
            ):
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim + enc_dim, hid_dim, 1, 
                batch_first=True, dropout=dropout)
        self.init_fc = nn.Linear(enc_dim, hid_dim)
        self.attn = nn.Linear(enc_dim, hid_dim)
        self.proj = nn.Linear(hid_dim, vocab_size)


    def attend(self, a, h):
        B, L, D = a.size()
        attn = Variable(torch.zeros(B, L))
        for i in range(L):
            attn[:,:,i] = self.attn(torch.cat(a, h))
        attn = F.softmax(attn.squeeze())
        return attn


    def forward(self, y, a, h):
        y = self.embedding(y)
        attn = self.attend(a, h)
        z = attn.unsqueeze(1).bmm(a).squeeze()
        output, h = self.rnn(torch.cat(y, z), h)
        output = self.proj(output)
        return output, h, attn


class Model(nn.Module):

    def __init__(self, 
            vocab_size, 
            emb_dim, 
            hid_dim, 
            dropout, 
            finetune
            ):
        nn.Module.__init__(self)

        modules = [m for m in models.vgg16(pretrained=True).features][:-2]
        self.encoder = nn.Sequential(*modules)
        if not finetune:
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        self.decoder = Decoder(
                vocab_size, 
                emb_dim, 
                512, 
                hid_dim, 
                dropout
                )


    def encode(self, img):
        a = self.encoder(img)
        a = a.view(a.size(0), 512, -1) # B x 512 x 196
        return a


    def init_dec_hidden(self, h, dim):
        return self.decoder.init_fc(h.mean(dim))


    def decode(self, y, a, h):
        return self.decoder(y, a, h)
        

