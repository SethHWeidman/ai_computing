"""Small helpers shared across the LLMs-from-scratch ch04 *reference* scripts.

The intent is to keep the per-topic reference scripts focused on the unique parts
(GQA/MLA/SWA/MoE) while sharing the boilerplate GPT scaffolding, KV-cache generation
loop, and the common transformer block structure.
"""

import typing

import torch
import torch.nn as nn

import attention_helpers


class PreNormTransformerBlock(nn.Module):
    """Standard pre-norm transformer block used in the reference scripts."""

    def __init__(
        self,
        att: nn.Module,
        ff: nn.Module,
        emb_dim: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        self.att = att
        self.ff = ff
        self.norm1 = attention_helpers.LayerNorm(emb_dim)
        self.norm2 = attention_helpers.LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTReferenceModel(nn.Module):
    """GPT-style model wrapper used across reference scripts."""

    def __init__(
        self,
        cfg: dict[str, typing.Any],
        block_factory: typing.Callable[[dict[str, typing.Any]], nn.Module],
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [block_factory(cfg) for _ in range(cfg["n_layers"])]
        )

        self.current_pos = 0

        self.final_norm = attention_helpers.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_ids = torch.arange(
            self.current_pos,
            self.current_pos + seq_len,
            device=in_idx.device,
            dtype=torch.long,
        )
        self.current_pos += seq_len
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for blk in self.trf_blocks:
            x = blk(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self) -> None:
        for blk in self.trf_blocks:
            att = getattr(blk, "att", None)
            reset = getattr(att, "reset_cache", None)
            if callable(reset):
                reset()
        self.current_pos = 0


def generate_text_simple_cached(
    model: GPTReferenceModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: typing.Optional[int] = None,
) -> torch.Tensor:
    """Greedy generation loop that always uses the model's KV-cache path."""

    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        model.reset_kv_cache()
        logits = model(idx[:, -ctx_len:])

        for _ in range(max_new_tokens):
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_idx], dim=1)
            logits = model(next_idx)

    return idx
