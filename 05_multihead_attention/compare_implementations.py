import torch
from torch import nn
import attention_helpers

# -----------------------------------------------------------------------------
# 1. The "Explainer" Implementation (List of Heads)
# -----------------------------------------------------------------------------


class SingleHeadAttention(nn.Module):
    """A single attention head that performs the full self-attention mechanism."""

    def __init__(
        self,
        d_in: int,
        d_head: int,
        context_length: int,
        # dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        # these are smaller linear layers, mapping d_in -> d_head
        self.W_query = nn.Linear(d_in, d_head, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_head, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_head, bias=qkv_bias)
        # non-essential step that improves model generalizability: adding dropout to
        # attention scores
        # self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        _, num_tokens, _ = input_vectors.shape

        # create representations of input `x`
        keys = self.W_key(input_vectors)
        queries = self.W_query(input_vectors)
        values = self.W_value(input_vectors)

        # 1. Raw attention scores
        # 2. Apply mask to ensure each
        # 3. Normalize (softmax) across rows
        attn_scores = queries @ keys.transpose(1, 2)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        scale = keys.shape[-1] ** -0.5
        attn_weights = torch.softmax(attn_scores * scale, dim=-1)
        # commented out for simplicity: here is where dropout would be applied
        # attn_weights = self.dropout(attn_weights)

        # `output_vectors` = weighted sum of values, weighted by attention weights
        output_vectors = attn_weights @ values
        return output_vectors


class MultiHeadAttentionExplainer(nn.Module):
    """MHA implemented as a list of independent SingleHeadAttention modules."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_attention_heads: int,
        qkv_bias: bool = False,
        # typically `dropout` would be included to help model generalizability; omitted
        # here for simplicity
        # dropout: float,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.num_attention_heads = num_attention_heads
        self.d_head = d_out // num_attention_heads

        # the list of independent heads
        self.attention_heads = nn.ModuleList(
            [
                SingleHeadAttention(
                    d_in=d_in,
                    d_head=self.d_head,
                    context_length=context_length,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_attention_heads)
            ]
        )

        # the final mixing projection
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        # step 1: compute self-attention within each head
        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(input_vectors))

        # step 2: concatenate head outputs
        concatenated_outputs = torch.cat(head_outputs, dim=-1)

        # step 3: mix them via a standard linear neural network operation
        return self.out_proj(concatenated_outputs)


# -----------------------------------------------------------------------------
# 2. The "Optimized" Implementation (Batched)
# -----------------------------------------------------------------------------


class MultiHeadAttentionOptimized(attention_helpers.MultiHeadAttentionBase):
    """The standard efficient implementation inheriting from the `MultiHeadAttentionBase`
    class."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        qkv_bias: bool = False,
        # dropout: float,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=0.0,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard implementation of MultiHeadAttention's `forward` method
        """
        # 1. create representations of each input vector
        queries, keys, values = self.project_qkv(x)
        # 2. create output vectors, including the concatentation of head outputs
        #    This is the function that FlashAttention will speed up
        context_vectors = self.compute_context_vectors(queries, keys, values)
        # 3. mix the head outputs
        return self.out_proj(context_vectors)


# -----------------------------------------------------------------------------
# 3. Verification Utilities
# -----------------------------------------------------------------------------


def transfer_weights(
    opt_model: MultiHeadAttentionOptimized, explainer_model: MultiHeadAttentionExplainer
) -> None:
    """
    Slices weights from the Optimized model and copies them to the Explainer model
    so they are mathematically identical.
    """
    with torch.no_grad():
        # 1. transfer output projection weights (direct copy)
        explainer_model.out_proj.weight.copy_(opt_model.out_proj.weight)
        explainer_model.out_proj.bias.copy_(opt_model.out_proj.bias)

        # 2. transfer q, k, v weights
        # the optimized model has shape (d_out, d_in)
        # we split this into 'num_heads' chunks of size (d_head, d_in)
        wq_chunks = opt_model.W_query.weight.chunk(explainer_model.num_heads, dim=0)
        wk_chunks = opt_model.W_key.weight.chunk(explainer_model.num_heads, dim=0)
        wv_chunks = opt_model.W_value.weight.chunk(explainer_model.num_heads, dim=0)

        for i, head in enumerate(explainer_model.heads):
            head.W_query.weight.copy_(wq_chunks[i])
            head.W_key.weight.copy_(wk_chunks[i])
            head.W_value.weight.copy_(wv_chunks[i])


def compare_implementations():
    torch.manual_seed(251228)

    # settings
    batch_size = 2
    context_length = 16
    d_in = 16
    d_out = 16
    num_heads = 4

    x = torch.randn(batch_size, context_length, d_in)

    # instantiate models
    opt_model = MultiHeadAttentionOptimized(d_in, d_out, context_length, num_heads)

    expl_model = MultiHeadAttentionExplainer(d_in, d_out, context_length, num_heads)

    # sync weights
    transfer_weights(opt_model, expl_model)

    # run forward passes
    opt_output = opt_model(x)
    expl_output = expl_model(x)

    # check results
    print(f"Optimized Output Shape: {opt_output.shape}")
    print(f"Explainer Output Shape: {expl_output.shape}")

    # check equality
    # we use a small tolerance for floating point arithmetic differences
    is_close = torch.allclose(opt_output, expl_output, atol=1e-6)

    if is_close:
        print("\nSUCCESS: The outputs match! (torch.allclose is True)")
        print(f"Max difference: {(opt_output - expl_output).abs().max().item()}")
    else:
        print("\nFAILURE: The outputs do not match.")
        print(f"Max difference: {(opt_output - expl_output).abs().max().item()}")


if __name__ == "__main__":
    compare_implementations()
