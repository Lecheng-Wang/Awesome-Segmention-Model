# encoding = utf-8

# @Author  ：Lecheng	Wang
# @Time	   : ${2023/9/20} ${TIME}
# @Function: Realization of	mobileNetv2	architecture

import math
import os
import torch
import torch.nn	as nn
from torchsummary import summary
from thop import profile

# Common 3×3 2D_Conv Layer
def	conv_3x3_bn(inputchannel, outputchannel, stride):
	return nn.Sequential(
		nn.Conv2d(inputchannel,	outputchannel, kernel_size=3, stride=stride, padding=1,	bias=False),
		nn.BatchNorm2d(outputchannel),
		nn.ReLU6(inplace=True)
	)
# Common 1×1 2D_Conv Layer
def	conv_1x1_bn(inputchannel, outputchannel):
	return nn.Sequential(
		nn.Conv2d(inputchannel,	outputchannel, kernel_size=1, stride=1,	padding=0, bias=False),
		nn.BatchNorm2d(outputchannel),
		nn.ReLU6(inplace=True)
	)

# The Single Block in MobileNetV2(expand_ratio aims	to expand channels of input_features)
class InvertedResidual(nn.Module):
	def	__init__(self, inputchannel, outputchannel,	stride,	expand_ratio):
		super(InvertedResidual,	self).__init__()
		expand_channels	= round(inputchannel * expand_ratio)
		self.use_res_connect = (stride == 1) and (inputchannel == outputchannel)
		if expand_ratio	== 1:
			self.conv =	nn.Sequential(
				nn.Conv2d(expand_channels, expand_channels,	kernel_size=3, stride=stride, padding=1, groups=expand_channels, bias=False),
				nn.BatchNorm2d(expand_channels),
				nn.ReLU6(inplace=True),
				nn.Conv2d(expand_channels, outputchannel, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(outputchannel),
			)
		else:
			self.conv =	nn.Sequential(
				nn.Conv2d(inputchannel,	expand_channels, kernel_size=1,	stride=1, padding=0, bias=False),
				nn.BatchNorm2d(expand_channels),
				nn.ReLU6(inplace=True),

				nn.Conv2d(expand_channels, expand_channels,	kernel_size=3, stride=stride, padding=1, groups=expand_channels, bias=False),
				nn.BatchNorm2d(expand_channels),
				nn.ReLU6(inplace=True),
				nn.Conv2d(expand_channels, outputchannel, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(outputchannel),
			)

	def	forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)

# Construct	the	MobileNetV2
class MobileNetV2(nn.Module):
	def	__init__(self, bands=3,	n_class=1280):
		super(MobileNetV2, self).__init__()
		Block		  =	InvertedResidual
		bands		  =	bands
		input_channel =	32
		last_channel  =	1280

		Block_Setting =	[
			[1,	16,	 1, 1],
			[6,	24,	 2, 2],
			[6,	32,	 3, 2],
			[6,	64,	 4, 2],
			[6,	96,	 3, 1],
			[6,	160, 3,	1],	 # 设置为2后下采样倍率为32, 设置为1后下采样率为16
			[6,	320, 1,	1],
		]
		self.features =	[conv_3x3_bn(bands,	input_channel, stride=2)]

		# t：Times of the Channels will be expanded in this Block group
		# c：OutputChannel of Features in this Block
		# n：Repeat times of	Block in this group
		# s：The	stride of GroupConv	in Block(Only first	Block need to set, aiming to Supsample the size	of Features)
		for	t, c, n, s in Block_Setting:
			output_channel = c
			for	i in range(n):
				if i ==	0:
					self.features.append(Block(input_channel, output_channel, stride=s,	expand_ratio=t))
				else:
					self.features.append(Block(input_channel, output_channel, stride=1,	expand_ratio=t))
				input_channel =	output_channel

		self.features.append(conv_1x1_bn(input_channel,	last_channel))
		self.features =	nn.Sequential(*self.features)

#		self.classifier = nn.Sequential(
#			nn.Dropout(0.2),
#			nn.Linear(last_channel, n_class),
#		)

		self._initialize_weights()

	def	forward(self, x):
		low_level_feature  = self.features[ : 4](x)
		high_level_feature = self.features[4:-1](low_level_feature)
		return low_level_feature, high_level_feature
#		x = self.features(x)
#		x = x.mean(3).mean(2)
#		x = self.classifier(x)
#		return x


	def	_initialize_weights(self):
		for	m in self.modules():
			if isinstance(m, nn.Conv2d):
				n =	m.kernel_size[0] * m.kernel_size[1]	* m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. /	n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n =	m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


def	mobilenetv2(bands=3, **kwargs):
	model =	MobileNetV2(bands=bands, **kwargs)
	return model

# Test Model Structure and Outputsize
if __name__	== "__main__":
	model		  =	mobilenetv2()
	device		  =	torch.device('cuda'	if torch.cuda.is_available() else 'cpu')
	model.to(device)
	x			  =	torch.randn(1, 3, 224, 224).to(device)	# Assume inputsize 3×224×224 RGB image
	print("Input shape:", x.shape)
	output		  =	model(x)
	flops, params =	profile(model, inputs=(x, ))
	print('flops: ', flops/1e9,	'params: ',	params/1e6)
	print("Low level output shape:", output[0].shape)
	print("High level output shape:", output[1].shape)
	summary(model, (3, 224,	224))