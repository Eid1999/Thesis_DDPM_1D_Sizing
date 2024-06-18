from requirements import *


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, heads=8):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, heads, batch_first=True)

    def forward(self, x):
        x = x[None].permute(1, 0, 2)
        output, _ = self.attention(x, x, x)
        output = output.permute(1, 0, 2).squeeze()
        return output


class SelfAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SelfAttention, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Linear transformation layers for query, key, and value
        self.query = nn.Linear(
            in_features=in_dim,
            out_features=hidden_dim,
            bias=False,
        )
        self.key = nn.Linear(
            in_features=in_dim,
            out_features=hidden_dim,
            bias=False,
        )
        self.value = nn.Linear(
            in_features=in_dim,
            out_features=hidden_dim,
            bias=False,
        )

        # Softmax function to compute attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Transform input into query, key, and value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, dtype=torch.float32)
        )

        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)

        # Weighted sum using attention weights
        attended_value = torch.matmul(attention_weights, value)

        return attended_value
