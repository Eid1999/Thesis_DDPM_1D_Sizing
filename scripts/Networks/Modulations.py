import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *


class Addition(nn.Module):

    def __init__(
        self,
        feature_size: int,
        input: int,
    ):
        super(Addition, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(
                feature_size,
                out_features=input,
            ),
            # nn.GELU(),
            # nn.Linear(
            #     feature_size * 4,
            #     out_features=input,
            # ),
        )

    def forward(
        self,
        y: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        beta = self.linear(y)

        return beta + time


class FiLM(Addition):

    def __init__(
        self,
        feature_size: int,
        input: int,
    ):
        super().__init__(feature_size, input * 2)

    def forward(
        self,
        y: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.linear(y).chunk(2, dim=-1)

        return gamma * time + beta


class Multiplication(Addition):

    def __init__(
        self,
        feature_size: int,
        input: int,
    ):
        super().__init__(feature_size, input)

    def forward(
        self,
        y: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.linear(y)

        return gamma * time
