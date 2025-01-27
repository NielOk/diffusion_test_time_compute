'''
Code for reverse diffuser and training.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple convolutional network for reverse diffusion
class SimpleConvNetDiffuser(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, base_channels=64):
        super(SimpleConvNetDiffuser, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
# Example usage
if __name__ == "__main__":
    # Input dimensions: batch_size x channels x height x width
    batch_size = 16
    x = torch.randn(batch_size, 5, 32, 32)  # 32x32x5 input

    model = SimpleConvNetDiffuser()
    output = model(x)
    print("Output shape:", output.shape)