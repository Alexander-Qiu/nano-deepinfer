from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 32
    ignore_eos: bool = False
    seed: int = 0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
