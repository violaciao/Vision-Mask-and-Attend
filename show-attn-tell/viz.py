import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_expand

import sys
import os

from trainer import Trainer
from config import Config


def denormalize_image(data, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]):
    '''
    denormalize images to range from 0 to 1
    - data: 4D numpy for tensor, [B, C, H, W]
    - mean: list of float, pixel mean of each channel
    - std: list of float, pixel standard deviation of each channel
    '''
    for c in range(3):
        data[:,c,:,:] = data[:,c,:,:] * std[c] + mean[c]
    data = np.clip(data, 0, 1)

    return data


def main():
    # set up config
    data_dir = sys.argv[1]
    ckpt_dir = sys.argv[2]
    config = Config(data_dir, ckpt_dir)
    config.batch_size = 1
    config.resume = True
    if not os.path.exists('plot'):
        os.mkdir('plot')

    # load best model
    trainer = Trainer(config)
    e, best_acc = trainer.load_ckpt(True)
    print('best acc = {:2.2f} @epoch {}'.format(best_acc, e))
    model = trainer.model
    dataloader = trainer.valid_dataloader
    classes = dataloader.dataset.classes

    viz_label = []
    for x, y in dataloader:
        y = y.numpy()[0]
        if y in viz_label:
            continue
        viz_label.append(y)
        
        # plot raw image
        raw_img = denormalize_image(x.numpy())
        raw_img = np.transpose(raw_img[0], (1, 2, 0))
        plt.subplot(1, config.repeat + 1, 1)
        plt.imshow(raw_img)
        plt.axis('off')

        x = Variable(x, volatile=True)
        y_input = Variable(torch.zeros(1, 1).long())
        a = model.encode(x)
        h = model.init_dec_hidden(a, 1)

        # attention at each step
        for step in range(2, config.repeat + 2):
            o, h, attn = model.decode(y_input, a, h)
            y_input = Variable(o.data.topk(1)[1].unsqueeze(0))
            cls = np.asscalar(y_input.data.numpy())
            plt.subplot(1, config.repeat + 1, step)
            plt.text(0, 1, classes[cls], color='black', 
                    backgroundcolor='white', fontsize=8)
            plt.imshow(raw_img)
            alphas = attn[0].data.numpy().reshape(14, 14)
            alpha_img = pyramid_expand(alphas, upscale=16, sigma=20)
            plt.imshow(alpha_img, alpha=0.85)
            plt.axis('off')

        plt.savefig('plot/sat_%d_%s.jpg' % (config.repeat, classes[y]), 
                bbox_inches='tight')
        plt.clf()

        if len(viz_label) == len(classes):
            break


if __name__ == '__main__':
    main()
