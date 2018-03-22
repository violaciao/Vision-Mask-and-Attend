import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import os

from config import get_config
from utils import denormalize, bounding_box, prepare_dirs
from dataset import get_test_loader, get_train_valid_loader
from trainer import Trainer


def get_locations_from_model(trainer, img):
    img_size = img.shape[1]
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

    for i in range(len(loc)):
        loc[i] = denormalize(img_size, loc[i])

    return loc


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
    
    for i, (x, y) in enumerate(dataloader):
        loc = get_locations_from_model(trainer, x)
        num_glimpses = len(loc)

        fig, axs = plt.subplots(nrows=1, ncols=num_glimpses)
        for j, ax in enumerate(axs.flat):
            ax.imshow(x[0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        for j, ax in enumerate(axs.flat):
            rect = bounding_box(
                    int(loc[j][0]), int(loc[j][1]), trainer.patch_size, 'r'
                    )
            ax.add_patch(rect)

        fig.tight_layout()
        plt.savefig(os.path.join(conf.viz_dir, 'viz_%d.jpg'%i), bbox_inches='tight')

        if i == 5:
            break


if __name__ == '__main__':
    main()
