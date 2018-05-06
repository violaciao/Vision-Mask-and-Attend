# Show Attend and Tell for Classification

This is a PyTorch implementation of visual self-attention, which combines [VGG](https://arxiv.org/abs/1409.1556) and [self-attention in Transformer](https://arxiv.org/abs/1706.03762).

## Requirements

* Python 3.6
* PyTorch 0.3
* MatPlotLib
* NumPy
* scikit-learn
* scikit-image

## Usage

1. Dataset is the same as descibed in [show-attn-tell](/show-attn-tell).

2. [Optional] Feel free to change the parameters in the file [config.py](./config.py).

3. Train the model: type ```python main.py <data_dir> <ckpt_dir> <mode>```.

## Model

* The features are extracted by [VGG net](https://arxiv.org/abs/1409.1556), with dimension 512 x 7 x 7.

* Three fully-connectedi(fc) layers are used for classification.

* [Self-attention](https://arxiv.org/abs/1706.03762) is added between feature and fc layers, if you want. Thus, we support three modes.
    - _vgg_: no self-attention is added
    - _mean_: takes the average along _length_ dimension before feeding into fc layers
    - _flatten_: flatten the extracted features before feeding int fc layers.

## Result

The table below reports the model size and the accuracy of flower dataset with different modes after more than 50 training epochs. With self-attention, the model size is six times smaller without losing the performance.

| mode | #params | accuracy(train/valid) | AUC | 
|:--------:|---------:|:----------:|:----:|
| _vgg_ | 136,360,773 | 93.35/95.38 | 0.9846 |
| _mean_ | 21,878,085 | 89.98/94.84 | 0.9912 |
| _flatten_ | 143,457,861 | 94.24/95.52 | 0.9268 |

