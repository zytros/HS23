from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = []
        layers_bn_down = []
        layers_pooling = []
        #TODO: raise NotImplementedError('Downsampling layers are not implemented!')
        
        for i in range(len(down_filter_sizes)):
            inp = 0
            if i == 0:
                inp = input_size
            else:
                inp = down_filter_sizes[i-1]
            layers_conv_down.append(
                nn.Conv2d(
                    in_channels=inp,
                    out_channels=down_filter_sizes[i],
                    kernel_size=kernel_sizes[i],
                    padding=conv_paddings[i]))
            
            layers_bn_down.append(
                nn.BatchNorm2d(
                    num_features=down_filter_sizes[i]))
            
            layers_pooling.append(
                nn.MaxPool2d(
                    kernel_size=pooling_kernel_sizes[i],
                    stride=pooling_strides[i],
                    return_indices=True))

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = []
        layers_bn_up = []
        layers_unpooling = []
        #TODO: raise NotImplementedError('Upsampling layers are not implemented!')
        
        for i in range(len(up_filter_sizes)):
            inp = 0
            if i == 0:
                inp = down_filter_sizes[-1]
            else:
                inp = up_filter_sizes[i-1]
            layers_unpooling.append(
                nn.MaxUnpool2d(
                    kernel_size=pooling_kernel_sizes[i],
                    stride=pooling_strides[i]))
            
            layers_conv_up.append(
                nn.Conv2d(
                    in_channels=inp,
                    out_channels=up_filter_sizes[i],
                    kernel_size=kernel_sizes[i],
                    padding=conv_paddings[i]))
            
            layers_bn_up.append(
                nn.BatchNorm2d(
                    num_features=up_filter_sizes[i]))

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        #TODO: raise NotImplementedError('Final convolution layer is not implemented!')
        
        self.conv_calsses = nn.Conv2d(in_channels=up_filter_sizes[-1], out_channels=11, kernel_size=1)

    def forward(self, x):
        #TODO: raise NotImplementedError('Forward function not implemented!')
        
        unpool_inds = []
        act_F = self.relu
        
        for i in range(len(self.layers_conv_down)):
            x = self.layers_conv_down[i](x)
            x = self.layers_bn_down[i](x)
            x = act_F(x)
            x, indices = self.layers_pooling[i](x)
            unpool_inds.append(indices)

        for i in range(len(self.layers_conv_up)):
            idx = (-i-1)
            x = self.layers_unpooling[i](x, unpool_inds[idx])
            x = self.layers_conv_up[i](x)
            x = self.layers_bn_up[i](x)
            x = act_F(x)

        x = self.conv_calsses(x)

        return x


def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
