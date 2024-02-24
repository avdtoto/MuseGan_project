from typing import List
from torch import nn, Tensor
import torch

def initialize_weights(layer: nn.Module, mean: float = 0.0, std: float = 0.02):
    """
    Initializes the weights of a layer to a normal distribution and biases to zero if applicable.

    Parameters:
    - layer (nn.Module): The layer to initialize.
    - mean (float): The mean of the normal distribution for weight initialization. Default is 0.0.
    - std (float): The standard deviation of the normal distribution for weight initialization. Default is 0.02.
    """
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d)):
        nn.init.normal_(layer.weight, mean, std)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

class Reshape(nn.Module):
    """
    A layer that reshapes its input tensor to a specified shape, maintaining the batch size.

    Parameters:
    - shape (List[int]): The target shape for the tensor, excluding the batch dimension.
    """

    def __init__(self, shape: List[int]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor to the specified shape.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The reshaped tensor.
        """
        return x.view(x.size(0), *self.shape)
