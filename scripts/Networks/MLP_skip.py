from requirements import *
from Networks.SinEmbPos import SinusoidalPosEmb
from Networks.attention import SelfAttention, MultiheadAttention
from Networks.MLP import MLP


class MLP_skip(MLP):

    def __init__(
        self,
        input_size=91,
        hidden_layers=[10],
        output_size=91,
        dim=6,
        y_dim=15,
        seed_value=0,
        attention_layers=[20],
        num_heads=[8],
    ):
        super().__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            dim=dim,
            y_dim=y_dim,
            seed_value=seed_value,
            attention_layers=attention_layers,
        )

    def forward(self, x, t, y=None):
        t = t.unsqueeze(dim=-1).type(torch.float).squeeze()
        X = {}

        for i, (
            time_embedding,
            y_mlp,
            hidden_layer,
        ) in enumerate(
            zip(
                self.time_embedding,
                self.y_mlps,
                self.hidden_layers,
            )
        ):
            t_aux = time_embedding(t)
            if y is not None:
                t_aux *= y_mlp(y)
            x = (
                hidden_layer(x + t_aux)
                if i - 1 < len(self.hidden_layers) // 2
                else hidden_layer(
                    x + X[f"{len(self.hidden_layers)-i-1}"] + t_aux,
                )
            )
            X[f"{i}"] = x
        return x
