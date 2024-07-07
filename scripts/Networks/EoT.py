import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from Networks.SinEmbPos import SinusoidalPosEmb
from Networks.attention import SelfAttention, MultiHeadAttention
from Networks import MLP
from Networks.Modulations import FiLM, Addition, Multiplication


class EoT(MLP):

    def __init__(
        self,
        input_size: int = 91,
        hidden_layers: list = [10],
        output_size: int = 91,
        dim: int = 6,
        y_dim: int = 15,
        seed_value: int = 0,
        num_heads: list = [8, 8],
        attention_layers: list = [10, 10],
    ):
        super().__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            dim=dim,
            y_dim=y_dim,
            seed_value=seed_value,
        )
        self.attention_layers = nn.ModuleList(
            [
                nn.Sequential(
                    MultiHeadAttention(
                        input_size if i == 0 else hidden_layers[i - 1],
                        num_heads=num_heads[i],
                    ),
                    nn.LayerNorm(input_size if i == 0 else hidden_layers[i - 1]),
                )
                for i in range(len(hidden_layers) + 1)
            ]
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=input_size if i == 0 else hidden_layers[i - 1],
                        out_features=(
                            hidden_layers[i] if i < len(hidden_layers) else output_size
                        ),
                    ),
                    nn.PReLU() if i < len(hidden_layers) else nn.Identity(),
                )
                for i in range(len(hidden_layers) + 1)
            ]
        )
        self.output_ln = nn.ModuleList(
            [
                (
                    nn.LayerNorm(hidden_layers[i])
                    if i < len(hidden_layers)
                    else nn.Identity()
                )
                for i in range(len(hidden_layers) + 1)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        t = t.unsqueeze(dim=-1).type(torch.float).squeeze()
        for i, (
            time_embedding,
            output_ln,
            hidden_layer,
            modulation,
            attention_layers,
        ) in enumerate(
            zip(
                self.time_embedding,
                self.output_ln,
                self.hidden_layers,
                self.modulation,
                self.attention_layers,
            )
        ):
            t_emb = time_embedding(t)
            if y is not None:
                t_emb = modulation(y, t_emb)
            x = attention_layers(x + t_emb)
            x = hidden_layer(x)
            x = output_ln(x)

        return x
