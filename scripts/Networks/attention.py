import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        head_dim: int = 37,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.num_heads = num_heads
        self.hidden_dim = head_dim * num_heads
        self.head_dim = head_dim

        assert (
            self.head_dim * num_heads == self.hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        # Linear transformation layers for query, key, and value
        self.query = nn.Linear(in_dim, self.hidden_dim, bias=False)
        self.key = nn.Linear(in_dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(in_dim, self.hidden_dim, bias=False)
        self.out = nn.Linear(self.hidden_dim, in_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.size()

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(batch_size, self.num_heads, self.head_dim)
        key = key.view(batch_size, self.num_heads, self.head_dim)
        value = value.view(batch_size, self.num_heads, self.head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )

        attention_weights = self.softmax(attention_scores)

        attended_values = torch.matmul(attention_weights, value)

        attended_values = attended_values.contiguous().view(batch_size, self.hidden_dim)
        output = self.out(attended_values)

        return output + x


class SelfAttention(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 200):
        super(SelfAttention, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Linear transformation layers for query, key, and value
        self.attention_matrices = nn.Linear(
            in_features=in_dim,
            out_features=hidden_dim * 3,
            bias=False,
        )
        self.out = nn.Linear(
            in_features=hidden_dim,
            out_features=in_dim,
            bias=False,
        )
        # Softmax function to compute attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> None:
        # Transform input into query, key, and value
        query, key, value = self.attention_matrices(x).chunk(3, dim=-1)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, dtype=torch.float32)
        )

        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)

        # Weighted sum using attention weights
        attended_value = torch.matmul(attention_weights, value)
        return self.out(attended_value) + x
