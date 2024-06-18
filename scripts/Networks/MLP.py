from requirements import *
from Networks.SinEmbPos import SinusoidalPosEmb
from Networks.attention import SelfAttention


class MLP(nn.Module):

    def __init__(
        self,
        input_size=91,
        hidden_layers=[10],
        output_size=91,
        dim=6,
        y_dim=15,
        seed_value=0,
        attention_layers=[10],
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
                        in_features=input_size if i == 0 else attention_layers[i - 1],
                        out_features=(
                            hidden_layers[i] if i < len(hidden_layers) else output_size
                        ),
                    ),
                    (
                        nn.LayerNorm(hidden_layers[i])
                        if i < len(hidden_layers)
                        else nn.Identity()
                    ),
                    (
                        SelfAttention(hidden_layers[i], attention_layers[i])
                        if i < len(hidden_layers)
                        else nn.Identity()
                    ),
                    nn.PReLU() if i < len(hidden_layers) else nn.Identity(),
                )
                for i in range(len(hidden_layers) + 1)
            ]
        )
        # sinu_pos_emb = SinusoidalPosEmb(dim)
        # sinu_y_embedding=SinusoidalPosEmb(y_dim)
        self.time_embedding = nn.ModuleList(
            [
                SinusoidalPosEmb(
                    dim=(input_size if i == 0 else attention_layers[i - 1]),
                )
                for i in range(len(hidden_layers) + 1)
            ]
        )
        self.y_vectorization = nn.ModuleList(
            [
                nn.Linear(
                    y_dim,
                    out_features=input_size if i == 0 else attention_layers[i - 1],
                )
                # )
                for i in range(len(hidden_layers) + 1)
            ]
        )

    def forward(self, x, t, y=None):
        t = t.unsqueeze(dim=-1).type(torch.float).squeeze()
        for time_embedding, y_vectorization, hidden_layer in zip(
            self.time_embedding, self.y_vectorization, self.hidden_layers
        ):
            t_aux = time_embedding(t)
            if y is not None:
                t_aux *= y_vectorization(y)
            x = hidden_layer(x + t_aux)
        return x
