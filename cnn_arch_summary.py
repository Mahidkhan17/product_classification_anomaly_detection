# Libraries
import torch
from torchsummary import summary
import cnn_arch

# CNN Model Configs/Params
in_channels = 1
out_channels = 3
in_dim = (1, 100, 100)

# CNN Instance
NET = cnn_arch.cnn_model(in_channels, out_channels).to('cpu')

# Print Summary
if(__name__ == '__main__'):
    summary(NET, in_dim)

