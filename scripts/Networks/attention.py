from requirements import *
from einops import rearrange, reduce, repeat
from math import lcm


class MultiheadAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        hidden_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        x = x[:, :, None]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b (h d) n")
        return out.squeeze()


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
