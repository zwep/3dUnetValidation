# encoding: utf-8

import inspect
from torch import nn as nn
import torch
import numpy as np


class Unet3D(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, chans=32, num_pool_layers=4, padding=0, **kwargs):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.final_activation_name = kwargs.get('final_activation', 'identity')
        self.debug = kwargs.get('debug', False)
        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock3D(in_chans, ch, ch * 2, padding=padding)])
        ch *= 2
        self.pooling_layer = nn.MaxPool3d(kernel_size=2, stride=2)  # Has no learn able parameters

        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock3D(ch, ch, ch * 2, padding=padding)]
            ch *= 2

        self.finest_conv = ConvBlock3D(ch, ch, ch * 2, padding=padding)
        ch *= 2

        self.up_sample_layers = nn.ModuleList()
        self.up_conv_layers = nn.ModuleList()
        for i in range(num_pool_layers):
            self.up_sample_layers += [nn.ConvTranspose3d(in_channels=ch, out_channels=ch, kernel_size=2,
                                                         padding=0, stride=2)]
            self.up_conv_layers += [ConvBlock3D(ch + ch // 2, ch // 2, ch // 2, padding=padding)]
            ch //= 2

        self.conv_final = nn.Conv3d(ch, out_chans, kernel_size=1, padding=0)
        self.final_activation = activation_selector(self.final_activation_name)

    def forward(self, input):
        stack = []
        output = input

        # Apply down-sampling layers
        counter = 0
        for layer in self.down_sample_layers:
            counter += 1
            output = layer(output)
            if self.debug:
                print('Downsampled layer ', output.shape)
            stack.append(output)
            output = self.pooling_layer(output)

        output = self.finest_conv(output)
        if self.debug:
            print('Finest conv ', output.shape)
        # Apply up-sampling layers
        counter = 0
        for up_layer, conv_layer in zip(self.up_sample_layers, self.up_conv_layers):
            downsample_layer = stack.pop()
            # Upscale
            output = up_layer(output)
            if self.debug:
                print('Upsample transpose ', output.shape)
            # Crop and copy
            down_shape = np.array(downsample_layer.shape[-3:])
            out_shape = np.array(output.shape[-3:])
            pos1 = np.array(down_shape) // 2 - np.array(out_shape) // 2
            pos2 = pos1 + np.array(out_shape)
            downsample_layer = downsample_layer[:, :, pos1[0]:pos2[0], pos1[1]:pos2[1], pos1[2]:pos2[2]]
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv_layer(output)
            if self.debug:
                print('Upsample conv ', output.shape)
            counter += 1

        output = self.conv_final(output)
        if self.debug:
            print('Final output ', output.shape)
        output_activation = self.final_activation(output)  # This is one of the changes I can make now..
        return output_activation


class ConvBlock3D(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation.
    """
    def __init__(self, in_chans, mid_chans, out_chans, kernel_size=3, padding=0):
        super().__init__()

        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(nn.Conv3d(in_chans, mid_chans, kernel_size=kernel_size, padding=padding),
                                    nn.BatchNorm3d(mid_chans),
                                    nn.ReLU(),
                                    nn.Conv3d(mid_chans, out_chans, kernel_size=kernel_size, padding=padding),
                                    nn.BatchNorm3d(out_chans),
                                    nn.ReLU())

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, mid_chans={self.mid_chans}, out_chans={self.out_chans})'


def activation_selector(activation_name, debug=False):
    # Used to easily plant some activation function
    # Normally contains a dictionary pointing to my own created activation functions
    if debug:
        print('Getting activation, ', activation_name)

    activation_name = activation_name.lower()
    nn_dict = {k.lower(): v for k, v in inspect.getmembers(torch.nn.modules.activation, inspect.isclass)}

    if activation_name in nn_dict.keys():
        output_actv_layer = nn_dict[activation_name]()
    else:
        output_actv_layer = None
        print('Unknown activation name ', activation_name)

    return output_actv_layer
