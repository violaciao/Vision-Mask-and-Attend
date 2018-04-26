import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm

import numpy as np
import os
import shutil
import random

from dataset import get_train_valid_loader, get_test_loader
from model import Model
from metrics import *


class Trainer(object):

    def __init__(self, config):
        self.config = config

        # load data
        self.train_dataloader, self.valid_dataloader = get_train_valid_loader(
                self.config.data_dir, self.config.batch_size, self.config.seed)
        num_channels = self.train_dataloader.dataset[0][0].shape[0]

        # define model
        self.model = Model(
                config.num_classes, 
                config.emb_dim, 
                config.hid_dim, 
                config.dropout, 
                config.finetune
                )
        if config.gpu:
            self.model.cuda()
        self.model_name = 'SAT_{}_{}_{}_{}'.format(
                config.emb_dim, config.hid_dim, config.repeat, config.finetune
            )

        # define optimizer and loss function
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.opt = optim.Adam(params, lr=config.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.summary


    @property
    def summary(self):
        print('[INFO] Train on %d samples, validate on %d samples' % (
            len(self.train_dataloader.dataset), len(self.valid_dataloader.dataset)))
        print('[INOF] Using model [%s]' % self.model_name)
        print('[INFO] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])
            ))


    def train(self):
        start_epoch, best_valid_acc = self.load_ckpt(self.config.load_best)

        for epoch in range(start_epoch, start_epoch + self.config.epochs):
            print('\n*** Epoch [%d] ***' % (epoch + 1))

            loss, cmat, auc = self.train_epoch(self.train_dataloader)
            acc = 100 * cmat.acc
            print('Training loss: {:2.4f}, acc: {:2.2f}'.format(loss, acc))
            #print(cmat)
            #print(auc)

            loss, cmat, auc = self.eval(self.valid_dataloader)
            acc = 100 * cmat.acc
            is_best = acc > best_valid_acc
            msg = 'Validation loss: {:2.4f}, acc: {:2.2f}'
            if is_best:
                best_valid_acc = acc
                msg += ' [*]'
            print(msg.format(loss, acc))
            #print(cmat)
            print(auc)

            self.save_ckpt(
                    {
                        'epoch': epoch + 1, 
                        'state_dict': self.model.cpu().state_dict(), 
                        'best_valid_acc': best_valid_acc 
                    }, is_best
                )

            if self.config.gpu:
                self.model.cuda()


    def convert_label(self, y):
        y = y.view(y.size(0), 1)# + 2
        y = y.repeat(1, self.config.repeat+2)
        #y[:,0] = 0     # <SOS>
        #y[:,-1] = 1    # <EOS>
        return y


    def train_epoch(self, dataloader):
        total_loss = 0
        num_classes = self.config.num_classes
        cmat = ConfusionMatrix(num_classes)
        auc = AUC(num_classes)
        self.model.train()

        for x, y in dataloader:
            self.opt.zero_grad()
            x, y = Variable(x), Variable(self.convert_label(y))
            B, seq_len = y.size()
            y_input = Variable(torch.zeros(B,1).long())
            if self.config.gpu:
                x, y = x.cuda(), y.cuda()
                y_input = y_input.cuda()

            a = self.model.encode(x)
            h = self.model.init_dec_hidden(a, 1)

            teacher_forcing = random.random() < self.config.teaching_ratio
            attns = []
            loss = 0
            if teacher_forcing:
                for i in range(0, seq_len):
                    o, h, attn = self.model.decode(y_input, a, h)
                    attns.append(attn)
                    loss += self.loss_fn(o, y[:,i])
                    y_input = y[:,i:i+1]
            else:
                for i in range(1, seq_len):
                    o, h, attn = self.model.decode(y_input, a, h)
                    attns.append(attn)
                    loss += self.loss_fn(o, y[:,i])
                    y_input = o.data.topk(1)[1]
                    y_input = Variable(y_input)

            loss /= float(seq_len)
            total_loss += np.asscalar(loss.data.cpu().numpy())
            y = y[:,-1]
            cmat.add(o.max(1)[1], y)
            auc.add(y, o)

            loss.backward()
            clip_grad_norm(self.model.parameters(), self.config.grad_clip)
            self.opt.step()

        return total_loss / len(dataloader), cmat, auc


    def eval(self, dataloader):
        total_loss = 0
        num_classes = self.config.num_classes
        cmat = ConfusionMatrix(num_classes)
        auc = AUC(num_classes)
        self.model.eval()

        for x, y in dataloader:
            x = Variable(x, volatile=True)
            y = Variable(self.convert_label(y), volatile=True)
            B, seq_len = y.size()
            y_input = Variable(torch.zeros(B,1).long())
            if self.config.gpu:
                x, y = x.cuda(), y.cuda()
                y_input = y_input.cuda()

            a = self.model.encode(x)
            h = self.model.init_dec_hidden(a, 1)

            attns = []
            loss = 0
            #majority = Majority()
            for i in range(1, seq_len):
                o, h, attn = self.model.decode(y_input, a, h)
                attns.append(attn)
                loss += self.loss_fn(o, y[:,i])
                y_input = o.data.topk(1)[1]
                #majority.add(y_input)
                y_input = Variable(y_input)

            loss /= float(seq_len)
            total_loss += np.asscalar(loss.data.cpu().numpy())
            y = y[:,-1]
            cmat.add(o.max(1)[1], y)
            #cmat.add(majority.vote_result(), y)
            auc.add(y, o)

        return total_loss / len(dataloader), cmat, auc


    def load_ckpt(self, best=False):
        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.config.ckpt_dir, filename)

        start_epoch = 0
        best_valid_acc = 0
        if self.config.resume and os.path.exists(ckpt_path):
            print("[INFO] Loading model from %s" % self.config.ckpt_dir)
            ckpt = torch.load(ckpt_path)

            # set values
            start_epoch = ckpt['epoch']
            best_valid_acc = ckpt['best_valid_acc']
            self.model.cpu().load_state_dict(ckpt['state_dict'])
            if self.config.gpu:
                self.model.cuda()

        return start_epoch, best_valid_acc


    def save_ckpt(self, state, is_best):
        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.config.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                    ckpt_path, os.path.join(self.config.ckpt_dir, filename))

