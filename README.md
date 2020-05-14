# Resnet Models
## Description 
Pytorch Implementation of ResNet34 and ResNet50 v1.5 on CIFAR-10 dataset. Original paper can be found [here](https://arxiv.org/pdf/1512.03385.pdf).
In original paper model is trained on ImageNet dataset but my implementation is based on CIFAR-10 dataset. Thus input size to my models has size 32*32*3.

## Models
ResNet is one of the models which revolutionized deep learning field. It is the first model which deep enough and was able to achive high accuracy on Image Classification task.
The problem with deep models was that they suffered from gradient vanishing problem, during backpropogation gradients of first layers were close to zero. 
Thus they were not able to learn anything at all. ResNet solves this problem with residual connection between blocks. Blocks are made from several convolutional layers.
There are 2 convolutions in ResNet34 whereareas in ResNet50 v1.5 there are 2 convolutions. Take a look at image below.  
<p align="center">
    <img src="./assets/block.png" width="220"/>
</p>    
ResNet composed of 4 main groups of blocks. The difference between ResNet34 and ResNet50 is that first one has 34 layers and later has 50 layers. Moreover 
the number blocks in group is different in each of them. The image below explains architecture of both models. Unlike Original paper, we have stride=1 in forst convolution 
of a model and don't use pooling after that. Moreover first group of blocks uses only stride=1. First blocks of later groups has stride=2.
This is done because input size of CIFAR-10 is 32x32x3 which is much smaller than size of ImageNet.
<p align="center">
    <img src="./assets/arch.jpg" width="220"/>
</p>

## Training
```
python3 train.py
```
SGD optmizer is used for training, with *learning_rate=0.01* *momentum=0.9* and *weight_decay=5e-4*. Training is done for 70 epochs on NVIDIA RTX2070.
**ResNet34**
<p align="center">
<img src = "assets/Accuracy_ResNet34.png" width ="220" /> <img src = "assets/Loss_ResNet34.png" width ="220" />
</p>
**ResNet50 v1.5**
<p align="center">
<img src = "assets/Accuracy_ResNet50.png" width ="220" /> <img src = "assets/Loss_ResNet50.png" width ="220" />
</p>

## Testing
```
python3 test.py
```
Model evaluation metrics are *TOP1 Accuracy* and *TOP5 Accuracy*.

## Requirements
* PyTorch 1.5.0
* Numpy 1.18.1
* Matplotlib 3.1.1
