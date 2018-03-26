import numpy as np
from utils import plot_images

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import os


data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(400),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'), 
            data_transforms['train'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'), 
            data_transforms['val'])

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(
            valid_data, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    name = 'test' if os.path.exists('test') else 'val'
    dataset = datasets.ImageFolder(
            os.path.join(data_dir, name), 
            data_transforms['val'])

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


if __name__ == '__main__':
    import sys, glob, os
    import random

    from_dir = sys.argv[1]
    to_dir = sys.argv[2]
    valid_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

    num_label = 17
    if not os.path.exists(to_dir):
        for i in range(num_label):
            os.system('mkdir -p %s' % os.path.join(to_dir, 'train', str(i)))
            os.system('mkdir -p %s' % os.path.join(to_dir, 'val', str(i)))

    files = glob.glob(os.path.join(from_dir, '*.jpg'))
    files = sorted(files)
    num_instance_per_label = 80
    n_valid = 0
    for i in range(len(files)):
        cls, n = str(i // num_instance_per_label), i % num_instance_per_label
        if n == 0:
            n_valid = 0

        name = files[i].split('/')[-1]
        if random.random() < valid_ratio and \
                n_valid < valid_ratio * num_instance_per_label:
            n_valid += 1
            os.system('cp %s %s' % (files[i], os.path.join(to_dir, 'val', cls, name)))
        elif n - n_valid < (1 - valid_ratio) * num_instance_per_label:
            os.system('cp %s %s' % (files[i], os.path.join(to_dir, 'train', cls, name)))
        else:
            n_valid += 1
            os.system('cp %s %s' % (files[i], os.path.join(to_dir, 'val', cls, name)))
