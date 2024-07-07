import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from Networks.SinEmbPos import SinusoidalPosEmb
from Networks.Modulations import FiLM, Addition, Multiplication


class MLP(nn.Module):

    def __init__(
        self,
        input_size: int = 91,
        hidden_layers: list = [10],
        output_size: int = 91,
        dim: int = 6,
        y_dim: int = 15,
        seed_value: int = 0,
    ):

        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        np.random.seed(seed_value)
        super(MLP, self).__init__()
        self.dim = dim

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=input_size if i == 0 else hidden_layers[i - 1],
                        out_features=(
                            hidden_layers[i] if i < len(hidden_layers) else output_size
                        ),
                    ),
                    (
                        nn.LayerNorm(hidden_layers[i])
                        if i < len(hidden_layers)
                        else nn.Identity()
                    ),
                    nn.PReLU() if i < len(hidden_layers) else nn.Identity(),
                )
                for i in range(len(hidden_layers) + 1)
            ]
        )
        self.time_embedding = nn.ModuleList(
            [
                nn.Sequential(
                    SinusoidalPosEmb(
                        dim=y_dim,
                    ),
                    nn.Linear(
                        y_dim,
                        out_features=y_dim * 4,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        y_dim * 4,
                        out_features=input_size if i == 0 else hidden_layers[i - 1],
                    ),
                )
                for i in range(len(hidden_layers) + 1)
            ]
        )

        self.modulation = nn.ModuleList(
            [
                FiLM(y_dim, input_size if i == 0 else hidden_layers[i - 1])
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
            hidden_layer,
            modulation,
        ) in enumerate(
            zip(
                self.time_embedding,
                self.hidden_layers,
                self.modulation,
            )
        ):
            t_emb = time_embedding(t)
            if y is not None:
                t_emb = modulation(y, t_emb)
            x = hidden_layer(x + t_emb)
        return x
