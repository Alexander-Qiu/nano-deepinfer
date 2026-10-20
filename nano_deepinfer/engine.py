"""A synchronous continuous-batching loop with a token budget per step."""

from collections import deque
from dataclasses import dataclass, field

import torch

from .cache import KVCache
from .model import Llama
from .sampling import SamplingParams, sample


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    params: SamplingParams
    rng: torch.Generator
    generated: list[int] = field(default_factory=list)
    cached: int = 0
    slot: int | None = None

    @property
    def tokens(self):
        return self.prompt + self.generated


@dataclass(frozen=True)
class Output:
    request_id: str
    token_ids: tuple[int, ...]
    finished: bool
    finish_reason: str | None


class Engine:
    def __init__(
        self, model: Llama, *, max_batch_size=4, max_seq_len=256,
        token_budget=64, eos_token_id: int | None = None,
    ):
        for name, value in (
            ("max_batch_size", max_batch_size), ("max_seq_len", max_seq_len),
            ("token_budget", token_budget),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_seq_len > model.config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the model context")
        if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("this version requires a CPU float32 model")
        self.model = model.eval()
        self.cache = KVCache(model.config, max_batch_size, max_seq_len)
        self.token_budget = token_budget
        self.eos_token_id = eos_token_id
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}
        self.last_step: tuple[dict, ...] = ()

    def add_request(self, request_id, token_ids, params=SamplingParams()):
        if request_id in self.requests:
            raise ValueError(f"duplicate active request: {request_id}")
        prompt = list(token_ids)
        if not prompt or any(
            not isinstance(t, int) or not 0 <= t < self.model.config.vocab_size for t in prompt
        ):
            raise ValueError("prompt must contain valid integer token IDs")
        if len(prompt) >= self.cache.max_seq_len:
            raise ValueError("prompt must leave room for at least one output token")
        request = Request(request_id, prompt, params, torch.Generator().manual_seed(params.seed))
        self.requests[request_id] = request
        self.waiting.append(request)

    @property
    def has_pending(self):
        return bool(self.requests)

    def _finish(self, request, reason):
        if request.slot is not None:
            self.cache.release(request.slot)
            self.running.remove(request)
        else:
            self.waiting.remove(request)
        del self.requests[request.request_id]
        return Output(request.request_id, tuple(request.generated), True, reason)

    def abort(self, request_id):
        return self._finish(self.requests[request_id], "aborted")

    @torch.inference_mode()
    def step(self) -> list[Output]:
        while self.waiting and self.cache.free_slots:
            request = self.waiting.popleft()
            request.slot = self.cache.allocate()
            self.running.append(request)

        # Decode first, then spend the remaining budget on prompt chunks.
        candidates = sorted(self.running, key=lambda r: not bool(r.generated))
        selected, chunks = [], []
        budget = self.token_budget
        for request in candidates:
            length = min(len(request.tokens) - request.cached, budget)
            if not length:
                break
            selected.append(request)
            chunks.append(request.tokens[request.cached:request.cached + length])
            budget -= length

        self.last_step = tuple(
            {"request_id": r.request_id, "phase": "decode" if r.generated else "prefill",
             "slot": r.slot, "start": r.cached, "tokens": len(chunk)}
            for r, chunk in zip(selected, chunks)
        )
        if not selected:
            return []
        lengths = [len(chunk) for chunk in chunks]
        token_ids = torch.zeros((len(selected), max(lengths)), dtype=torch.long)
        for row, chunk in enumerate(chunks):
            token_ids[row, :len(chunk)] = torch.tensor(chunk)
        logits = self.model(
            token_ids, self.cache, [r.slot for r in selected],
            [r.cached for r in selected], lengths,
        )
        outputs = []
        for row, (request, length) in enumerate(zip(selected, lengths)):
            request.cached += length
            if request.cached == len(request.tokens):
                token = sample(logits[row, length - 1], request.params, request.rng)
                request.generated.append(token)
                reason = None
                if token == self.eos_token_id and not request.params.ignore_eos:
                    reason = "eos"
                elif len(request.generated) >= request.params.max_tokens:
                    reason = "length"
                elif len(request.tokens) >= self.cache.max_seq_len:
                    reason = "context"
                outputs.append(
                    self._finish(request, reason) if reason else
                    Output(request.request_id, tuple(request.generated), False, None)
                )
            # Rotate serviced requests so small budgets do not starve other decodes.
            if request in self.running:
                self.running.remove(request)
                self.running.append(request)
        return outputs

    def generate(self, prompts, params=SamplingParams()):
        """Blocking convenience API. Results follow input order; engine must be idle."""
        if self.has_pending:
            raise RuntimeError("use step() while requests are active")
        ids = []
        try:
            for i, prompt in enumerate(prompts):
                request_id = str(i)
                self.add_request(request_id, prompt, params)
                ids.append(request_id)
        except Exception:
            for request_id in ids:
                self.abort(request_id)
            raise
        completed = {}
        while self.has_pending:
            for output in self.step():
                if output.finished:
                    completed[output.request_id] = output
        return [completed[request_id] for request_id in ids]
