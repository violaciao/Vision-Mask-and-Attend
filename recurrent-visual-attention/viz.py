import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from config import get_config
from utils import denormalize, bounding_box, prepare_dirs
from dataset2 import get_test_loader, get_train_valid_loader
from trainer import Trainer


def get_locations_from_model(trainer, img):
    '''
    Forward trainer to get locations at each glimpse

    Args:
        - trainer: Trainer object, whose model has been trained
        - img: 4D tensor, [B, C, H, W]

    Returns:
        - loc: list of (x,y) tuples, locations from raw image
            (num_glimpses, num_stacks, 2)
    '''
    img_size = img.shape[-1]
    if trainer.use_gpu:
        img = img.cuda()
    img = Variable(img, volatile=True)

    h_t, l_t = trainer.reset()
    loc = [[l.squeeze().data.cpu().numpy() for l in l_t]]

    for t in range(trainer.num_glimpses - 1):
        h_t, l_t, b_t, p = trainer.model(img, l_t, h_t)
        loc.append([l.squeeze().data.cpu().numpy() for l in l_t])

    h_t, l_t, b_t, log_p, p = trainer.model(img, l_t, h_t, True)
    loc.append([l.squeeze().data.cpu().numpy() for l in l_t])

    for i in range(len(loc)):
        for j in range(trainer.num_stacks):
            size = img_size // 2 ** j
            loc[i][j] = denormalize(size, loc[i][j])

    return loc


def denormalize_image(data, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]):
    '''
    denormalize images to range from 0 to 1

    - data: 4D numpy for tensor, [B, H, W, C]
    - mean: list of float, pixel mean of each channel
    - std: list of float, pixel standard deviation of each channel
    '''
    for c in range(3):
        data[:,:,:,c] = data[:,:,:,c] * std[c] + mean[c]
    data = np.clip(data, 0, 1)

    return data


def main():
    conf = get_config()[0]
    prepare_dirs(conf)

    conf.is_train = False
    kwargs = {}
    if conf.use_gpu:
        kwargs = {'num_workers': 1, 'pin_memory': True}

    batch_size = 1
    dataloader = get_test_loader(conf.data_dir, batch_size, **kwargs)

    trainer = Trainer(conf, dataloader)
    setattr(trainer, 'batch_size', batch_size)
    trainer.load_checkpoint(True)
    
    viz_label = []
    for i, (x, y) in enumerate(dataloader):
        y = y.numpy()[0]
        if y in viz_label:
            continue
        viz_label.append(y)

        loc = get_locations_from_model(trainer, x)
        imgs = [x]
        for j in range(1, trainer.num_stacks):
            imgs.append(F.avg_pool2d(Variable(x), 2**j).data)

        if x.size(-1) != trainer.num_channels:
            for j in range(0, trainer.num_stacks):
                imgs[j] = imgs[j].transpose(1, 3)
                imgs[j] = denormalize_image(imgs[j].numpy())

        num_glimpses = len(loc)
        n_row = 2 ** trainer.num_stacks - 1
        n_col = 2 ** (trainer.num_stacks - 1) * num_glimpses

        fig = plt.figure(1, (n_col, n_row))
        gridspec.GridSpec(n_row, n_col)
        row_start_idx = [0]
        for j in range(1, trainer.num_stacks):
            idx = 2 ** (trainer.num_stacks - j) + row_start_idx[j-1]
            row_start_idx.append(idx)

        for j in range(num_glimpses):
            for k in range(trainer.num_stacks):
                r = row_start_idx[k]
                c = 2 ** (trainer.num_stacks - 1) * j
                sz = 2 ** (trainer.num_stacks - k - 1)
                ax = plt.subplot2grid((n_row, n_col), (r, c), colspan=sz, rowspan=sz)
                ax.imshow(imgs[k][0])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                x_coord, y_coord = int(loc[j][k][0]), int(loc[j][k][1])
                for s in range(trainer.num_patches):
                    size = trainer.patch_size * 2 ** s
                    rect = bounding_box(x_coord, y_coord, size, 'r')
                    ax.add_patch(rect)

        fig.tight_layout()
        plt.savefig(os.path.join(conf.viz_dir, 'viz_%d.jpg' % len(viz_label)), 
                bbox_inches='tight')

        if len(viz_label) == 3:
            break


if __name__ == '__main__':
    main()
