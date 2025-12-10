"""Shared attention utilities used across demo scripts."""

import typing

import torch
from torch import nn


class MultiHeadAttentionBase(nn.Module):
    """Shared base class for multi-head attention demos."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project inputs to per-head Q, K, V with head-wise layout."""

        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        return queries, keys, values

    def _compute_context_vectors(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        use_mask: bool = True,
    ) -> torch.Tensor:
        """Apply scaled dot-product attention using the shared helper."""

        return compute_context_vectors(
            queries, keys, values, self.mask, self.dropout, use_mask=use_mask
        )


def compute_context_vectors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: typing.Optional[torch.Tensor],
    dropout: nn.Module,
    use_mask: bool = True,
) -> torch.Tensor:
    """Apply the standard scaled dot-product attention block.

    Args:
        queries, keys, values: Shape (b, num_heads, num_tokens, head_dim)
        mask: Optional causal mask of shape (context_len, context_len)
        dropout: Dropout module applied to attention weights
        use_mask: Whether to apply the provided mask

    Returns:
        Tensor of shape (b, num_tokens, num_heads * head_dim)
    """

    b, num_heads, num_tokens, head_dim = queries.shape
    attn_scores = queries @ keys.transpose(-2, -1)

    if mask is not None and use_mask:
        mask_bool = mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, float("-inf"))

    scale = head_dim**-0.5
    attn_weights = torch.softmax(attn_scores * scale, dim=-1)
    attn_weights = dropout(attn_weights)

    context_vectors = attn_weights @ values
    context_vectors = (
        context_vectors.transpose(1, 2)
        .contiguous()
        .view(b, num_tokens, num_heads * head_dim)
    )
    return context_vectors


class LayerNorm(nn.Module):
    """Simple LayerNorm used across GPT-style blocks."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU activation used in feed-forward blocks."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """Standard 2-layer FFN block used in transformer layers."""

    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__ == "__main__":
    # Simple demo of input/output shapes for MultiHeadAttentionBase
    batch_size = 2
    num_tokens = 5
    d_in = 16
    d_out = 16
    num_heads = 4
    context_length = 8

    x = torch.randn(batch_size, num_tokens, d_in)
    mha = MultiHeadAttentionBase(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=True,
    )

    print(f"Input x shape: {x.shape}  # (batch_size, num_tokens, d_in)")
    queries, keys, values = mha._project_qkv(x)
    print(
        f"Q/K/V shape: {queries.shape}  # (batch_size, num_heads, num_tokens, head_dim)"
    )

    context = mha._compute_context_vectors(queries, keys, values, use_mask=True)
    print(
        f"Context vectors shape: {context.shape}  # (batch_size, num_tokens, num_heads "
        "* head_dim"
    )
