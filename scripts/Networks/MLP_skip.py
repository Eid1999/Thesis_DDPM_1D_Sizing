import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from libraries import *
from Networks.SinEmbPos import SinusoidalPosEmb
from Networks.attention import SelfAttention, MultiHeadAttention
from Networks.MLP import MLP


class MLP_skip(MLP):

    def __init__(
        self,
        input_size: int = 91,
        hidden_layers: list = [10],
        output_size: int = 91,
        dim: int = 6,
        y_dim: int = 15,
        seed_value: int = 0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            dim=dim,
            y_dim=y_dim,
            seed_value=seed_value,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        t = t.unsqueeze(dim=-1).type(torch.float).squeeze()
        X = {}

        for i, (
            time_embedding,
            hidden_layer,
            film_attention,
        ) in enumerate(
            zip(
                self.time_embedding,
                self.hidden_layers,
                self.FiLM,
            )
        ):
            t_emb = time_embedding(t)
            if y is not None:
                t_emb = film_attention(y, t_emb)
            x = (
                hidden_layer(x + t_emb)
                if i - 1 < len(self.hidden_layers) // 2
                else hidden_layer(
                    x + X[f"{len(self.hidden_layers)-i-1}"] + t_emb,
                )
            )
            X[f"{i}"] = x
        return x
