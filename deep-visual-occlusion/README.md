# Deep Vision Occlusion

This is a **PyTorch** implementation of [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)   
by *Matthew D Zeiler and Rob Fergus*.  

Credits -   
[MarkoArsenovic](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases), [Deep Visual Recognition (cs231n)](http://cs231n.github.io/transfer-learning/), [PyTorch.org](http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) and [PyTorch Git](https://github.com/pytorch/examples/tree/master/imagenet).

## Requirements

- python 3.5+
- pytorch 0.3+
- tensorboard_logger

## Usage

1. Download dataset (eg. [flower](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) ) and process the data with **data_struct_\*.py**;
2. Run the model with **model.py** `[options]`; <br />
Alternatively, to train model with Tensorboard, use **\*model_tensorboard.py** `[options]`; <br />
3. Visualize Occlusion Experiments with **occlusion.py**.

## Results

We have tested our model on various of datasets: 1) a 5-classes subset of the flower dataset; 2) brain scan images; 3) pathology dataset. Following are some of the performance results.

| Data | Model | Batch size | Validation Accuracy |
|:--------:|:---------:|:----------:|:----------:|
flower_5 | ResNet18 | 10 | 90.35%
flower_5 | ResNet18 | 20 | **95.92%**
brain_T1 | ResNet18 | 10 | 78.4%  
brain_T1_FL | ResNet18 | 10 | 57.72%  
brain_T1_GD | ResNet18 | 10 | 81.66% 
brain_T2 | ResNet18 | 10 | 65.88% 
brain_T2_FL | ResNet18 | 10 | 65.66% 
brain_MIX | ResNet18 | 10 | **71.14%**  

**NB:**  
For flower_5 dataset, the best model reaches accuracy 95.92%. <br />
For brain CT dataset, the best performance is 81.66% on T1_GD modality, and 71.14% for ensemble.  


## Occlusion Experiment

Occlusion experiments for producing the heat maps that show visually the influence of each region on the classification.

### Usage

Produce the heat map and plot with  **occlusion.py** and store the visualizations in ```--output_dir```:
 
 `python3 occlusion.py [options]`
 
### Visualization Examples on Resnet18
![daisy](https://github.com/violaciao/Vision-Mask-and-Attend/blob/master/convolutional-translearn/Results/daisy/daisy_1_m.png)
*daisy - original, size 20 stride 10*

![dandelion](https://github.com/violaciao/Vision-Mask-and-Attend/blob/master/convolutional-translearn/Results/dandelion/dandelion_1_m.png)
*dandelion - original, size 20 stride 10*
