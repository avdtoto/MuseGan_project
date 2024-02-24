import torch
from torch import nn
from .utils import Reshape

class BarGenerator(nn.Module):
    """
    A generator model for creating musical bars.

    Attributes:
    - z_dimension (int): Dimensionality of the noise vector. Default is 32.
    - hid_features (int): Number of features in hidden layers. Default is 1024.
    - hid_channels (int): Number of channels in hidden convolutional layers. Default is 512.
    - out_channels (int): Number of channels in the output layer. Default is 1.
    - n_steps_per_bar (int): Number of time steps per bar in the output. Default is 16.
    - n_pitches (int): Number of pitches per time step in the output. Default is 84.
    """

    def __init__(self, z_dimension=32, hid_features=1024, hid_channels=512, out_channels=1, n_steps_per_bar=16, n_pitches=84):
        super(BarGenerator, self).__init__()
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches

        self.net = nn.Sequential(
            nn.Linear(4 * z_dimension, hid_features),
            nn.BatchNorm1d(hid_features),
            nn.ReLU(inplace=True),
            Reshape(shape=[hid_channels, hid_features // hid_channels, 1]),
            self._conv_transpose_block(hid_channels, hid_channels, (2, 1), (2, 1)),
            self._conv_transpose_block(hid_channels, hid_channels // 2, (2, 1), (2, 1)),
            self._conv_transpose_block(hid_channels // 2, hid_channels // 2, (2, 1), (2, 1)),
            self._conv_transpose_block(hid_channels // 2, hid_channels // 2, (1, 7), (1, 7)),
            nn.ConvTranspose2d(hid_channels // 2, out_channels, kernel_size=(1, 12), stride=(1, 12)),
            Reshape(shape=[1, 1, self.n_steps_per_bar, self.n_pitches])
        )

    def _conv_transpose_block(self, in_channels, out_channels, kernel_size, stride, padding=0):
        """
        Helper function to create a block of ConvTranspose2d -> BatchNorm2d -> ReLU layers.

        Parameters:
        - in_channels (int): Number of channels in the input.
        - out_channels (int): Number of channels produced by the convolution.
        - kernel_size (tuple): Size of the convolving kernel.
        - stride (tuple): Stride of the convolution.
        - padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output tensor.
        """
        return self.net(x)
