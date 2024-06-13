from requirements import *


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

        # Define hidden layer sizes (you can adjust these)
        super(Simulator, self).__init__()
        # Create a list of linear layers for hidden layers
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=input_size if i == 0 else hidden_layers[i],
                    out_features=(
                        hidden_layers[i + 1]
                        if i + 1 < len(hidden_layers)
                        else output_size
                    ),
                )
                for i in range(len(hidden_layers))
            ]
        )
        self.activation_layer = nn.ReLU()

        # Output layer

    def forward(self, x):
        # Pass through hidden layers with activation function (e.g., ReLU)
        for i, layer in enumerate(self.hidden_layers):
            x = (
                self.activation_layer(layer(x))
                if i < len(self.hidden_layers)
                else layer(x)
            )

        # Output layer without activation (depends on loss function)
        return x
