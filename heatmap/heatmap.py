# -*- coding:utf-8 -*-
import os
import sys
import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import argparse
from PIL import Image

import torch
from torch.autograd import Function
from torchvision import models
from torchvision import utils
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torchvision.transforms import Normalize

sys.path.append("../")

# from BCNN_ResNet101 import Net
# import BCNN_ResNet101

from BCNN_VGG16 import Net
import BCNN_VGG16

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

class FeatureExtractor():
	""" Class for extracting activations and
	registering gradients from targetted intermediate layers """
	def __init__(self, model, target_layers):
		self.model = model.features
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []

		for name, module in self.model._modules.items():
			x = module(x)
			if name in self.target_layers:
				x.register_hook(self.save_gradient)
				outputs += [x]

		x = x.view(1, 512, 7**2)
		x = torch.bmm(x, torch.transpose(x, 1, 2)) / (7**2)
		x = x.view(1, 512**2)
		x = torch.sqrt(x + 1e-5)                # L2
		x = nn.functional.normalize(x)

		if "bmm" in self.target_layers:
			x.register_hook(self.save_gradient)
			outputs += [x]

		return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output = self.feature_extractor(x)
		# print(output.shape)
		output = output.view(x.size(0), -1)
		output1, output2, output3, output4, output5, output6 = self.model.classifier(output)
		return target_activations, output1, output2, output3, output4, output5, output6

def preprocess_image(img):
	# means=[0.5,0.5,0.5]
	# stds=[0.5,0.5,0.5]
	means = [0.485, 0.456, 0.406]
	stds = [0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img.requires_grad_()
	return input

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))
	print("Grad-cam image saved.")

class GradCam:
	def __init__(self, model, target_layer_names, label_number):
		self.model = model
		self.model.eval()
		self.model = model.cuda()
		self.label_number = label_number

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		features, output1, output2, output3, output4, output5, output6 = self.extractor(input.cuda())
		outputs = [output1, output2, output3, output4, output5, output6]
		output = outputs[self.label_number][0]
		# print(output.shape)

		if index == None:
			index = output.data.max(1, keepdim=True)[1]

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)

		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot.requires_grad_()
		# print one_hot.cuda() * output1
		one_hot = torch.sum(one_hot.cuda() * output)

		# print one_hot, "asdas" #预测类别的score
		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		self.model.zero_grad()
		one_hot.backward() #retain_variables=True

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		# print(grads_val.shape) #在target_layers后的图片尺寸 1*512*14*14

		target = features[-1]
		# print features[0].shape, features[-1].shape
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.ones(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image-path', type=str, default='./128_3_2_4_4_3_3165.jpg', help='Input image path')
	parser.add_argument('--label-num', type=int, default='1', help='Which label you wanna know. From 0 to 5.')
	args = parser.parse_args()

	print("Using GPU for acceleration")

	return args

def read_img(filename):
	# img = cv2.imread(args.image_path)
	# img = np.float32(cv2.resize(img, (224, 224))) / 255
	# input = preprocess_image(img)

	input = Image.open(filename)

	crop_size = (224, 224)
	input_transform = Compose([Resize(crop_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	input = input_transform(input)

	return input

if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args()

	device_ids = [0]
	device = torch.device("cuda")

	# Can work with any model, but it assumes that the model has a
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.

	model = Net()
	# model.load_state_dict(torch.load('../model/BCNN_resnet101_epoch_100.pth').module.state_dict())
	model.load_state_dict(torch.load('../model/BCNN_vgg16_epoch_100.pth').state_dict())
	model = model.to(device)
	# model.eval()
	# print(model)
	summary(model, (3, 224, 224))

	# grad_cam = GradCam(model, target_layer_names = ["avgpool"], label_number = args.label_num)  #default=35
	grad_cam = GradCam(model, target_layer_names = ["28"], label_number = args.label_num)  #default=35

	input = read_img(args.image_path)

	# If None, returns the map for the highest scoring category.
	# Otherwise, targets the requested index.
	target_index = 0
	mask = grad_cam(input, target_index)
	show_cam_on_image(img, mask)