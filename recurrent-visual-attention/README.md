# Recurrent Visual Attention

This is a **PyTorch** implementation of [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) by *Volodymyr Mnih, Nicolas Heess, Alex Graves and Koray Kavukcuoglu*. 

Folked from [kevinzakka](https://github.com/kevinzakka/recurrent-visual-attention). The original ```readme``` file is [here](original_README.md).

## Requirements

- python 3.5+
- pytorch 0.3+
- tensorboard_logger

## Modifications

* Generalized to other datasets (gray-scale and RGB)
* Input images do not need to be square
* Fixed some bugs, like tensor shape of image arrays in a minibatch
