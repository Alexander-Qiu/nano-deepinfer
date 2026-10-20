import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x, positions, theta):
    dim = x.shape[-1]
    freq = theta ** (-torch.arange(0, dim, 2, dtype=x.dtype) / dim)
    angles = positions[:, None, :, None] * freq
    angles = torch.cat((angles, angles), dim=-1)
    rotated = torch.cat((-x[..., dim // 2:], x[..., :dim // 2]), dim=-1)
    return x * angles.cos() + rotated * angles.sin()


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.head_dim = dim // config.num_attention_heads
        kv_dim = config.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, positions, cache, layer, slots, starts, lengths):
        batch, width, _ = x.shape
        c = self.config
        q = self.q_proj(x).view(batch, width, c.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, width, c.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, width, c.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        k, v = cache.update(layer, slots, starts, lengths, k, v)
        repeat = c.num_attention_heads // c.num_key_value_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        key_positions = torch.arange(k.shape[2])[None, None, None, :]
        ends = torch.tensor(starts) + torch.tensor(lengths)
        allowed = (key_positions <= positions[:, None, :, None]) & (
            key_positions < ends[:, None, None, None]
        )
        # Absolute positions are essential when q has one token and k has a prefix.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        out = (weights @ v).transpose(1, 2).reshape(batch, width, c.hidden_size)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, *args):
        x = x + self.self_attn(self.input_layernorm(x), *args)
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        if (
            config.model_type != "llama" or config.rope_scaling
            or config.attention_bias or config.mlp_bias or config.hidden_act != "silu"
            or getattr(config, "rope_interleaved", False)
            or getattr(config, "sliding_window", None)
            or config.hidden_size % config.num_attention_heads
            or config.num_attention_heads % config.num_key_value_heads
            or head_dim % 2 or getattr(config, "head_dim", head_dim) != head_dim
        ):
            raise ValueError("requires standard Llama with plain RoPE and no biases")
        self.config = config
        self.model = Decoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.inference_mode()
    def forward(self, token_ids, cache, slots, starts, lengths):
        """Return [batch, padded_query_length, vocab]; ignore padded query logits."""
        positions = torch.tensor(starts)[:, None] + torch.arange(token_ids.shape[1])[None, :]
        x = self.model.embed_tokens(token_ids)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, positions, cache, i, slots, starts, lengths)
        return self.lm_head(self.model.norm(x))
