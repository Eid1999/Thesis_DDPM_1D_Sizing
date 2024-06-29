import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *


class Simulator(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers=[10],
    ):
        seed_value = 0
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(0)

        super(Simulator, self).__init__()

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=input_size if i == 0 else hidden_layers[i - 1],
                        out_features=(
                            hidden_layers[i] if i < len(hidden_layers) else output_size
                        ),
                    ),
                    # (
                    #     nn.BatchNorm1d(hidden_layers[i])
                    #     if i < len(hidden_layers)
                    #     else nn.Identity()
                    # ),
                    nn.PReLU() if i < len(hidden_layers) else nn.Identity(),
                )
                for i in range(len(hidden_layers) + 1)
            ]
        )

    def forward(self, x):

        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)

        return x
