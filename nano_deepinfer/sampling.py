"""Sampling state belongs to a request, never to the batch."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0
    ignore_eos: bool = False

    def __post_init__(self):
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer (0 disables it)")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")


def sample(logits: torch.Tensor, params: SamplingParams, rng: torch.Generator) -> int:
    if not torch.isfinite(logits).all():
        raise ValueError("model returned non-finite logits")
    if params.temperature == 0:
        return int(logits.argmax())
    # Center before scaling so tiny positive temperatures cannot overflow the maximum.
    scores = logits.double()
    scores = (scores - scores.max()) / params.temperature
    # Sort once: top-k and nucleus filtering then share the same ordering.
    scores, indices = scores.sort(descending=True)
    if params.top_k:
        scores = scores[:params.top_k]
        indices = indices[:params.top_k]
    probs = scores.softmax(dim=-1)
    # Keep the token that crosses top_p, including at least one token.
    probs = probs.masked_fill(probs.cumsum(-1) - probs >= params.top_p, 0)
    return int(indices[torch.multinomial(probs, 1, generator=rng)])
