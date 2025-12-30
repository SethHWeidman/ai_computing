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
        num_heads: int,
        dropout: float = 0.0,
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

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project inputs to per-head Q, K, V with head-wise layout.

        Returns (queries, keys, values), each with shape (batch, num_heads, num_tokens,
        head_dim).
        """

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

    def compute_context_vectors(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        use_mask: bool = True,
    ) -> torch.Tensor:
        """Apply scaled dot-product attention and return flattened outputs."""

        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout=self.dropout,
            mask=(self.mask if use_mask else None),
        )


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    dropout: nn.Module,
    return_per_head: bool = False,
    mask: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention with an optional causal mask.

    If `return_per_head` is True, returns context vectors with shape `(batch, num_heads,
    num_tokens, head_dim)`; otherwise returns the flattened output shape `(batch,
    num_tokens, num_heads * head_dim)`.
    The former shape, that is returned if `return_per_head` is True,
    """

    b, num_heads, num_tokens, head_dim = queries.shape
    attn_scores = queries @ keys.transpose(-2, -1)

    if mask is not None:
        mask_bool = mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

    scale = head_dim**-0.5
    attn_weights = torch.softmax(attn_scores * scale, dim=-1)
    attn_weights = dropout(attn_weights)

    context_vectors = attn_weights @ values
    if return_per_head:
        return context_vectors

    # Concatenate heads: transpose to (b, num_tokens, num_heads, head_dim), then make
    # memory contiguous so per-token head slices sit next to each other, and finally
    # flatten (num_heads, head_dim) -> d_out to get (b, num_tokens, d_out).
    concatenated_head_outputs = (
        context_vectors.transpose(1, 2)
        .contiguous()
        .view(b, num_tokens, num_heads * head_dim)
    )

    return concatenated_head_outputs


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
        num_heads=num_heads,
        dropout=0.0,
        qkv_bias=True,
    )

    print(f"Input x shape: {x.shape}  # (batch_size, num_tokens, d_in)")
    queries, keys, values = mha.project_qkv(x)
    print(
        f"Q/K/V shape: {queries.shape}  # (batch_size, num_heads, num_tokens, head_dim)"
    )

    context = mha.compute_context_vectors(queries, keys, values)
    print(
        f"Context vectors shape: {context.shape}  # (batch_size, num_tokens, num_heads "
        "* head_dim"
    )
