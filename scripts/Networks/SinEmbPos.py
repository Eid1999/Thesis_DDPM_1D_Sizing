import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        try:
            emb = x[:, None] * emb[None, :]
        except IndexError:
            emb = x[None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 != 0:
            ##padding for the odd dimesion
            emb = torch.cat(
                (emb, torch.zeros(emb.shape[0], device="cuda")[:, None]), dim=-1
            )

        return emb
