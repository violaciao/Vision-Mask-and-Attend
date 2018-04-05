""" 
Models
"""

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.init as init

import numpy as np


"""
resnet18, resnet34, resnet50, resnet101, resnet152
densenet121, densenet161, densenet169, densenet201
squeezenet1_0, squeezenet1_1
alexnet,
inception_v3,
vgg11, vgg13, vgg16, vgg19
vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


- If you want to freeze the layers by its name
```
for name, params in model_conv.named_parameters():
    if name =! "something":
        params.requires_grad=False
```

- If you want to freeze the first few layers
```
model_ft = models.resnet50(pretrained=True)
ct = 0
for child in model_ft.children():
ct += 1
if ct < 7:
    for param in child.parameters():
        param.requires_grad = False
```
ct = []
for name, child in model_conv.named_children():
    if "layer1" in ct:
        execute this
    ct.append(name)
"""

def all_pretrained_models(n_class, use_gpu=True, freeze_layers=False, freeze_initial_layers=False, name="resnet18", pretrained=True):
    if pretrained:
        weights = "imagenet"
    else:
        weights = False
    if name == "alexnet":
        print("[Building alexnet]")
        model_conv = models.alexnet(pretrained=weights)
    elif name == "inception_v3":
        print("[Building inception_v3]")
        model_conv = models.inception_v3(pretrained=weights)
    elif name == "resnet18":
        print("[Building resnet18]")
        model_conv = models.resnet18(pretrained=weights)
    elif name == "resnet34":
        print("[Building resnet34]")
        model_conv = models.resnet34(pretrained=weights)
    elif name == "resnet50":
        print("[Building resnet50]")
        model_conv = models.resnet50(pretrained=weights)
    elif name == "resnet101":
        print("[Building resnet101]")
        model_conv = models.resnet101(pretrained=weights)
    elif name == "resnet152":
        print("[Building resnet152]")
        model_conv = models.resnet152(pretrained=weights)
    elif name == "densenet121":
        print("[Building densenet121]")
        model_conv = models.densenet121(pretrained=weights)
    elif name == "densenet169":
        print("[Building densenet169]")
        model_conv = models.densenet169(pretrained=weights)
    elif name == "densenet201":
        print("[Building densenet201]")
        model_conv = models.densenet201(pretrained=weights)
    elif name == "squeezenet1_0":
        print("[Building squeezenet1_0]")
        model_conv = models.squeezenet1_0(pretrained=weights)
    elif name == "squeezenet1_1":
        print("[Building squeezenet1_1]")
        model_conv = models.squeezenet1_1(pretrained=weights)
    elif name == "vgg11":
        print("[Building vgg11]")
        model_conv = models.vgg11(pretrained=weights)
    elif name == "vgg13":
        print("[Building vgg13]")
        model_conv = models.vgg13(pretrained=weights)
    elif name == "vgg16":
        print("[Building vgg16]")
        model_conv = models.vgg16(pretrained=weights)
    elif name == "vgg19":
        print("[Building vgg19]")
        model_conv = models.vgg19(pretrained=weights)
    elif name == "vgg11_bn":
        print("[Building vgg11_bn]")
        model_conv = models.vgg11_bn(pretrained=weights)
    elif name == "vgg13_bn":
        print("[Building vgg13_bn]")
        model_conv = models.vgg13_bn(pretrained=weights)
    elif name == "vgg16_bn":
        print("[Building vgg16_bn]")
        model_conv = models.vgg16_bn(pretrained=weights)
    elif name == "vgg19_bn":
        print("[Building vgg19_bn]")
        model_conv = models.vgg19_bn(pretrained=weights)
    else:
        raise ValueError

    if not pretrained:
        print("[Initializing the weights randomly........]")
        if "densenet" in name:
            num_ftrs = model_conv.classifier.in_features
            model_conv.classifier = nn.Linear(num_ftrs, n_class)
        elif "squeezenet" in name:
            in_ftrs = model_conv.classifier[1].in_channels
            out_ftrs = model_conv.classifier[1].out_channels
            features = list(model_conv.classifier.children())
            features[1] = nn.Conv2d(in_ftrs, n_class, kernel_size=(2, 2), stride=(1, 1))
            features[3] = nn.AvgPool2d(12, stride=1)
            model_conv.classifier = nn.Sequential(*features)
            model_conv.num_classes = n_class
        elif "vgg" in name or "alexnet" in name:
            print("[Building VGG or Alexnet classifier]")
            num_ftrs = model_conv.classifier[6].in_features
            features = list(model_conv.classifier.children())[:-1]
            features.extend([nn.Linear(num_ftrs, n_class)])
            model_conv.classifier = nn.Sequential(*features)
        elif "vgg" in name or "alexnet" in name:
            print("[Building VGG or Alexnet classifier]")
            num_ftrs = model_conv.classifier[6].in_features
            features = list(model_conv.classifier.children())[:-1]
            features.extend([nn.Linear(num_ftrs, n_class)])
            model_conv.classifier = nn.Sequential(*features)
        else:
            print("[Building inception_v3 or Resnet]")
            num_ftrs = model_conv.fc.in_features
            model_conv.fc = nn.Linear(num_ftrs, n_class)


    else:
        if freeze_layers:
            for i, param in model_conv.named_parameters():
                param.requires_grad = False
        else:
            print("[All layers will be trained]")
        # Parameters of newly constructed modules have requires_grad=True by default
        if "densenet" in name:
            print("[Building Densenet]")
            num_ftrs = model_conv.classifier.in_features
            model_conv.classifier = nn.Linear(num_ftrs, n_class)
            if freeze_initial_layers:
                print("[Densenet: Freeezing layers only till denseblock1 including]")
                ct = []
                for name, child in model_conv.features.named_children():
                    if "denseblock1" in ct: #freeze all layers from layer1 inclusive
                        for params in child.parameters():
                            params.requires_grad = True
                    ct.append(name)
        elif "squeezenet" in name:
            print("[Building Squeezenet]")
            in_ftrs = model_conv.classifier[1].in_channels
            out_ftrs = model_conv.classifier[1].out_channels
            features = list(model_conv.classifier.children())
            features[1] = nn.Conv2d(in_ftrs, n_class, kernel_size=(2, 2), stride=(1, 1))
            features[3] = nn.AvgPool2d(12, stride=1)
            model_conv.classifier = nn.Sequential(*features)
            model_conv.num_classes = n_class
            if freeze_layers:
                ct = []
                print("[Squeezenet: Freeezing layers only till denseblock1 including]")
                for name, child in model_conv.features.named_children():
                    if "3" in ct:
                        for params in child.parameters():
                            params.requires_grad = True
                    ct.append(name)

        elif "vgg" in name or "alexnet" in name:
            print("[Building VGG or Alexnet classifier]")
            num_ftrs = model_conv.classifier[6].in_features
            features = list(model_conv.classifier.children())[:-1]
            features.extend([nn.Linear(num_ftrs, n_class)])
            model_conv.classifier = nn.Sequential(*features)
            if freeze_layers:
                ct = []
                print("[Alex or VGG: Freeezing layers only till denseblock1 including]")
                for name, child in model_conv.features.named_children():
                    if "5" in ct:
                        for params in child.parameters():
                            params.requires_grad = True
                    ct.append(name)

        elif "resnext" in name:
            print("[Building resnext]")
            num_ftrs = model_conv.last_linear.in_features
            model_conv.last_linear = nn.Linear(num_ftrs, n_class)
            if freeze_layers:
                ct = []
                for name, child in model_conv.features.named_children():
                    if "4" in ct:
                        for param in child.parameters():
                            param.requires_grad = True
                    ct.append(name)

        else:
            print("[Building inception_v3 or Resnet]")
            num_ftrs = model_conv.fc.in_features
            model_conv.fc = nn.Linear(num_ftrs, n_class)

            if freeze_initial_layers:
                if "resnet" in name:
                    print("[Resnet: Freezing layers only till layer1 including]")
                    ct = []
                    for name, child in model_conv.named_children():
                        if "layer1" in ct:
                            for params in child.parameters():
                                params.requires_grad = True
                        ct.append(name)
                else:
                    print("[Inception: Freezing layers only till layer1 including]")
                    ct = []
                    for name, child in model_conv.named_children():
                        if "Conv2d_4a_3x3" in ct:
                            for params in child.parameters():
                                params.requires_grad = True
                        ct.append(name)

    if use_gpu:
        model_conv = model_conv.cuda()
    else:
        model_conv = model_conv
    return model_conv