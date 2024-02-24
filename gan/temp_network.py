import torch
from torch import nn, Tensor
from .utils import Reshape

class TemporalNetwork(nn.Module):
    """
    A network designed to process and generate temporal features from a noise vector,
    useful in applications requiring understanding or generation of data over time,
    such as music.

    Parameters:
    - z_dimension (int): The dimensionality of the input noise vector.
    - hid_channels (int): The number of hidden channels in the convolutional layers.
    - n_bars (int): The number of bars (or temporal segments) to generate.

    The network employs transposed convolutions to transform the input noise vector
    into a series of temporal features across specified 'n_bars'.
    """

    def __init__(self, z_dimension=32, hid_channels=1024, n_bars=2):
        super(TemporalNetwork, self).__init__()
        self.net = nn.Sequential(
            Reshape(shape=[z_dimension, 1, 1]),
            nn.ConvTranspose2d(z_dimension, hid_channels, kernel_size=(2, 1), stride=(1, 1)),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hid_channels, z_dimension, kernel_size=(n_bars - 1, 1), stride=(1, 1)),
            nn.BatchNorm2d(z_dimension),
            nn.ReLU(inplace=True),
            Reshape(shape=[z_dimension, n_bars]),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the TemporalNetwork, transforming the input noise vector
        into temporal features.

        Parameters:
        - x (Tensor): The input noise vector with shape (batch_size, z_dimension).

        Returns:
        - Tensor: The generated temporal features with shape (batch_size, z_dimension, n_bars).
        """
        return self.net(x)
