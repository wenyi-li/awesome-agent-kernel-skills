# This is an example model.py
# It implements a custom CUDA kernel for element-wise addition
# IMPORTANT: Do not change this file!
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return self.alpha * a + b


def get_inputs():
    a = torch.randn(1, 128)
    b = torch.randn(1, 128)
    return [a, b]


def get_init_inputs():
    return [2.0]