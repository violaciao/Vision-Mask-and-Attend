import torch
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
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
        - conv_loc: list f (x, y) tuples, locations from conved image, 
                whose size is half of the raw image
    '''
    img_size = img.shape[-1]
    if trainer.use_gpu:
        img = img.cuda()
    img = Variable(img, volatile=True)

    h_t, l_t = trainer.reset()
    loc = [l_t.squeeze().data.cpu().numpy()]

    for t in range(trainer.num_glimpses - 1):
        h_t, l_t, b_t, p = trainer.model(img, l_t, h_t)
        loc.append(l_t.squeeze().data.cpu().numpy())

    h_t, l_t, b_t, log_p, p = trainer.model(img, l_t, h_t, True)
    loc.append(l_t.squeeze().data.cpu().numpy())

    conv_loc = []
    for i in range(len(loc)):
        conv_loc.append(denormalize(img_size/2, loc[i]))
        loc[i] = denormalize(img_size, loc[i])

    return loc, conv_loc


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

        loc, conv_loc = get_locations_from_model(trainer, x)
        x_conv = F.max_pool2d(
                trainer.model.sensor.conv(Variable(x, volatile=True)), 2)
        x_conv = x_conv.data
        if x.size(-1) != trainer.num_channels:
            x = x.transpose(1,3)
            x_conv = x_conv.transpose(1,3)

        denormalize_image(x.numpy())
        denormalize_image(x_conv.numpy())
        num_glimpses = len(loc)

        fig, (axs1, axs2) = plt.subplots(nrows=2, ncols=num_glimpses)

        # raw image
        for j, ax in enumerate(axs1.flat):
            ax.imshow(x[0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        for j, ax in enumerate(axs1.flat):
            x, y = int(loc[j][0]), int(loc[j][1])
            for s in range(trainer.num_patches):
                size = trainer.patch_size * 2 ** s
                rect = bounding_box(x, y, size, 'r')
                ax.add_patch(rect)

        # conved image
        for j, ax in enumerate(axs2.flat):
            ax.imshow(x_conv[0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        for j, ax in enumerate(axs2.flat):
            x, y = int(conv_loc[j][0]), int(conv_loc[j][1])
            for s in range(trainer.num_patches):
                size = trainer.patch_size * 2 ** s
                rect = bounding_box(x, y, size, 'r')
                ax.add_patch(rect)

        fig.tight_layout()
        plt.savefig(os.path.join(conf.viz_dir, 'viz_%d.jpg' % len(viz_label)), 
                bbox_inches='tight')

        if len(viz_label) == 3:
            break


if __name__ == '__main__':
    main()
