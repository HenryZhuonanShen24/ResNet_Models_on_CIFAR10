import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.pylab as plt2
import os

from resnet50 import ResNet50
from resnet34 import ResNet34

#Check GPU, connect to it if it is available 
device = ''
if torch.cuda.is_available():
	device = 'cuda'
	print("CUDA is available. GPU will be used for testing.")
else:
	device = 'cpu'


BEST_ACCURACY = 0

# Preparing Data
print("==> Prepairing data ...")
#transformation on validation data
transform_validation = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

#Download Validation data and apply transformation
validation_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_validation)

#Put data into loader, specify batch_size
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=100, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Function to show CIFAR images
def show_data(image):
	plt.imshow(np.transpose(image[0], (1, 2, 0)), interpolation='bicubic')
	plt.show()


model = ResNet50()
#model = ResNet34()
model = model.to(device)
print("Upload the model ...")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
model.load_state_dict(torch.load('./checkpoint/resnet50.pth'))
model.eval()


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	
	res = []
	for k in topk:
	    correct_k = correct[:k].view(-1).float().sum(0)
	    res.append(correct_k)
	return res

def test():
	model.eval()
	with torch.no_grad():
		accuracy1 = 0
		accuracy5 = 0
		for x,y in validation_loader:
			x, y = x.to(device), y.to(device)
			model.eval()
			yhat = model(x)
			yhat = yhat.reshape(-1, 10)
			a1, a5 = accuracy(yhat, y, topk=(1,5))
			accuracy1 += a1 
			accuracy5 += a5

		return (accuracy1/len(validation_data)).item(), (accuracy5/(len(validation_data))).item()


acc1, acc5 = test()

print("--------------------------")
print("|       ResNet50         |")
print("--------------------------")
print("| TOP1 Accuracy:", format(100*acc1, '.4f'), "|")
print("| TOP5 Accuracy:", format(100*acc5, '.4f'), "|")
print("--------------------------\n")
