import numpy as np
from utils import plot_images

import torch
from torch.utils.data import Dataset, DataLoader


class FlowerDataset(Dataset):

    def __init__(self, data):
        self.data = torch.stack([torch.FloatTensor(d[0]) for d in data])
        self.label = torch.LongTensor([d[1] for d in data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]
        

def load_data(data_dir, valid_size, random_seed):
    import glob, os, random, scipy.misc

    files = glob.glob(os.path.join(data_dir, '*.jpg'))
    files.sort()

    num_labels = 17
    num_per_label = 80
    split = int(num_per_label*valid_size)
    train_data = []
    valid_data = []
    random.seed(random_seed)
    H = 0
    W = 0

    for i in range(num_labels):
        idx = list(range(num_per_label))
        random.shuffle(idx)

        for j in idx[split:]:
            img = scipy.misc.imread(files[i*num_per_label+j]) / 255.0
            train_data.append([img, i])
            H = max(H, img.shape[0])
            W = max(W, img.shape[1])

        for j in idx[:split]:
            img = scipy.misc.imread(files[i*num_per_label+j]) / 255.0
            valid_data.append([img, i])
            H = max(H, img.shape[0])
            W = max(W, img.shape[1])

    # pad images to the same dimension
    def __pad(x):
        for i in range(len(x)):
            h_pad = W - x[i][0].shape[1]
            left_pad = h_pad // 2
            right_pad = h_pad - left_pad
            v_pad = H - x[i][0].shape[0]
            top_pad = v_pad // 2
            bottom_pad = v_pad - top_pad
            x[i][0] = np.pad(x[i][0], 
                    ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), 
                    mode='constant', constant_values=(0, 0))

    __pad(train_data)
    __pad(valid_data)

    return train_data, valid_data


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

    train_data, valid_data = load_data(data_dir, valid_size, random_seed)
    train_dataset = FlowerDataset(train_data)
    valid_dataset = FlowerDataset(valid_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

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
    data, _ = load_data(data_dir, 0, 0)
    dataset = FlowerDataset(data)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
