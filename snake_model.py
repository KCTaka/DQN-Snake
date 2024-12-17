import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from helper import shape_after_conv
import numpy as np
import re

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_shape, output_size):
        super(FCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.relu = nn.ReLU()
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_shape[0]))
        for i in range(1, len(hidden_shape)):
            hidden_layer = nn.Linear(hidden_shape[i-1], hidden_shape[i])
            self.hidden_layers.append(hidden_layer)
        
        self.output_layer = nn.Linear(hidden_shape[-1], output_size)
        
    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.relu(self.hidden_layers[i](x))
        
        x = self.output_layer(x)
        return x
    
    def test(self):
        random_input = torch.rand(1, self.input_size)
        x = self.forward(random_input)
        if x.shape[1] != self.output_size:
            print("Model is not working correctly")
            print("Expected output shape: ", (1, self.output_size))
            print("Output shape: ", x.shape)
            return False
        print("Input shape: ", random_input.shape)
        print("Output shape: ", x.shape)
        print("Model is working correctly")
        return True
    
class CNN(nn.Module):
    def __init__(self, input_shape, **cnn_structure):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.cnn_structure = cnn_structure
        
        self.relu = nn.ReLU()
        
        self.layer_types = list(cnn_structure.keys())
        self.layer_params = list(cnn_structure.values())
        
        self.hidden_layers = nn.ModuleList()
        for i, (layer_type, layer_param) in enumerate(zip(self.layer_types, self.layer_params)):
            if re.match(r"conv_and_relu\d+", layer_type):
                in_channels, out_channels, kernel_size, stride, padding = layer_param.values()
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                self.hidden_layers.append(conv_layer)
            elif re.match(r"fc\d+", layer_type):
                in_features, out_features = layer_param.values()
                fc_layer = nn.Linear(in_features, out_features)
                self.hidden_layers.append(fc_layer)
            elif re.match(r"max_pool\d+", layer_type):
                kernel_size, stride, padding = layer_param.values()
                max_pool = nn.MaxPool2d(kernel_size, stride, padding)
                self.hidden_layers.append(max_pool)
            
        
    def forward(self, x):
        for  i, layer_type in enumerate(self.layer_types):
            if re.match(r"conv_and_relu\d+", layer_type):
                x = self.relu(self.hidden_layers[i](x))
            if re.match(r"fc\d+", layer_type):
                x = x.view(x.size(0), -1)
                x = self.hidden_layers[i](x)
            if re.match(r"max_pool\d+", layer_type):
                x = self.hidden_layers[i](x)
            
        return x
    
    def test_shape(self):
        x_shape = np.array(self.input_shape)
        
        print(self.layer_types)
        print(self.layer_params)
        
        for i, (layer_type, layer_param) in enumerate(zip(self.layer_types, self.layer_params)):
            if re.match(r"conv_and_relu\d+", layer_type):
                in_channels, out_channels, kernel_size, stride, padding = layer_param.values()
                input_shape = x_shape
                
                output_shape = np.zeros_like(x_shape)
                output_shape[1:] = shape_after_conv(input_shape[1:], kernel_size, stride, padding)
                output_shape[0] = out_channels
                
                print(f"Layer {i+1}: Conv2d({in_channels}, {out_channels}, {kernel_size}, {stride}, {padding})")
                print(f"\tInput {input_shape} -> Output {output_shape}")
                
                x_shape = output_shape
                
            if re.match(r"fc\d+", layer_type):
                in_features, out_features = layer_param.values()
                input_shape = np.prod(x_shape)
                output_shape = out_features
                
                print(f"Layer {i+1}: Linear({in_features}, {out_features})")
                print(f"\tInput {input_shape} -> Output {output_shape}")
                
                x_shape = out_features
                
            if re.match(r"max_pool\d+", layer_type):
                kernel_size, stride, padding = layer_param.values()
                input_shape = x_shape
                
                output_shape = np.zeros_like(x_shape)
                output_shape[1:] = shape_after_conv(input_shape[1:], kernel_size, stride, padding)
                output_shape[0] = out_channels
                
                print(f"Layer {i+1}: MaxPool2d({kernel_size}, {stride}, {padding})")
                print(f"\tInput {input_shape} -> Output {output_shape}")
                
                x_shape = output_shape
                
    def test(self, batch_size=5):
        random_input = torch.rand(batch_size, *self.input_shape)
        x = self.forward(random_input)
        
        print("Input shape: ", random_input.shape)
        print("Output shape: ", x.shape)
        print("Model is working correctly")
        return True
    
class MultiCNN(nn.Module):
    def __init__(self, input_shape, output_size, num_frames, state_space, **cnn_structure):
        super(MultiCNN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_size
        self.num_frames = num_frames
        self.state_space = state_space
        self.cnn_structure = cnn_structure
        
        self.CNN_model = CNN(input_shape, **cnn_structure)
        self.linear1 = nn.Linear(num_frames*state_space, output_size)
        self.linear2 = nn.Linear(output_size, output_size)
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
            
        batch_size = x.size(0)
        x = x.view(-1, *self.input_shape)
        x = self.CNN_model(x)
        x = x.view(batch_size, -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
    
    def test(self, batch_size=5):
        x = torch.rand(batch_size, self.num_frames, *self.input_shape)
        # x = torch.rand(self.num_frames, *self.input_shape)
        print("Input shape: ", x.shape)
        y = self.forward(x)
        print("Output shape: ", y.shape)
        
            

if __name__ == "__main__":
    model = FCNN(6, [16, 3], 3)
    model.test()
    
    # kwargs = {
    #     "conv_and_relu1": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "conv_and_relu2": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "max_pool1": {"kernel_size": 2, "stride": 2, "padding": 0},
    #     "conv_and_relu3": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "conv_and_relu4": {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "max_pool2": {"kernel_size": 2, "stride": 2, "padding": 0},
    #     "conv_and_relu5": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "conv_and_relu6": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "max_pool3": {"kernel_size": 2, "stride": 2, "padding": 0},
    #     "conv_and_relu7": {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "conv_and_relu8": {"in_channels": 512, "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "max_pool4": {"kernel_size": 2, "stride": 2, "padding": 0},
    #     "conv_and_relu9": {"in_channels": 512, "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "conv_and_relu10": {"in_channels": 512, "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "max_pool5": {"kernel_size": 2, "stride": 2, "padding": 0},
    #     "fc1": {"in_features": 7*7*512, "out_features": 4096},
    #     "fc2": {"in_features": 4096, "out_features": 1000},
    # }
    
    # kwargs = {
    #     "conv_and_relu1": {"in_channels": 2, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "max_pool1": {"kernel_size": 2, "stride": 2, "padding": 0},
    #     "conv_and_relu2": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
    #     "max_pool2": {"kernel_size": 2, "stride": 2, "padding": 0},
    #     "fc1": {"in_features": 32*5*5, "out_features": 128},
    #     "fc2": {"in_features": 128, "out_features": 10},
    # }
    
    kwargs = {
        "conv_and_relu1": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 4, "padding": 0},
        "conv_and_relu2": {"in_channels": 16, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 0},
        "fc1": {"in_features": 32*9*9, "out_features": 256},
        "fc2": {"in_features": 256, "out_features": 3},
    }
    
    model = CNN(input_shape=(4, 84, 84), **kwargs)
    model.test_shape()
    model.test(batch_size=64)
    
    # model = MultiCNN(input_shape=(2, 20, 20), output_size=3, num_frames=5, state_space=10, **kwargs)
    # model.test(batch_size=64)