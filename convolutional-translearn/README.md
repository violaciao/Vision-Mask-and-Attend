# Deep Vision Recognition

This is a **PyTorch** implementation of [Deep Visual Recognition - Transfer Learning](http://cs231n.github.io/transfer-learning/).   

Credits -   
http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html and https://github.com/pytorch/examples/tree/master/imagenet.

## Requirements

- python 3.5+
- pytorch 0.3+
- tensorboard_logger

## Usage

1. Download dataset (e.g., [flower](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) )

2. Run the following command to process data  
```Python
python data_struct_*.py
```
3. Train the model  
```Python
python fine_tuning_model.py [options]
```

4. Tensorboard
```Python
python fine_tuning_model_tensorboard.py [options]
```

## Results

We have tried the subset of the flower dataset, which has 5 classes. Following are the hyperparamters we used for flower dataset. Others are set as default.

| data | model | batch size | validation accuracy |
|:--------:|:---------:|:----------:|:----------:|
flower_5 | ResNet18 | 10 | 90.35%
flower_5 | ResNet18 | 20 | **95.92%**
brain_T1 | ResNet18 | 10 | 78.4%  
brain_T1_FL | ResNet18 | 10 | 57.72%  
brain_T1_GD | ResNet18 | 10 | 81.66% 
brain_T2 | ResNet18 | 10 | 65.88% 
brain_T2_FL | ResNet18 | 10 | 65.66% 
brain_MIX | ResNet18 | 10 | **71.14%**  


For flower_5 dataset, the best model reaches accuracy 95.92%. <br />
For brain CT dataset, the best performance is 81.66% on T1_GD modality, and 71.14% for ensemble.

---

# Occlusion Experiment

Occlusion experiments for producing the heat maps that show visually the influence of each region on the classification.  
The original README file is [here](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases/blob/master/README.md).

### Usage:

Produce the heat map and plot with  **occlusion.py** and store the visualizations in ```--output_dir```:
 
 `python3 occlusion.py [options]`
 
 ### Visualization Examples on Resnet18:
![daisy](https://github.com/violaciao/Vision-Mask-and-Attend/blob/master/convolutional-translearn/Results/daisy/daisy_1_m.png)
*daisy - original, size 20 stride 10*

![dandelion](https://github.com/violaciao/Vision-Mask-and-Attend/blob/master/convolutional-translearn/Results/dandelion/dandelion_1_m.png)
*dandelion - original, size 20 stride 10*
