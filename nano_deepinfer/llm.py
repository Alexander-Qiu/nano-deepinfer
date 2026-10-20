from .engine.model_runner import ModelRunner
from .engine.scheduler import Scheduler
from .engine.sequence import Sequence
from .loading import load_pretrained
from .sampling_params import SamplingParams


class LLM:
    def __init__(
        self, model, *, max_num_seqs=4, max_model_len=256,
        max_num_batched_tokens=64, revision=None,
    ):
        model, self.tokenizer = load_pretrained(model, revision)
        if max_model_len > model.config.max_position_embeddings:
            raise ValueError("max_model_len exceeds the model context")
        self.vocab_size = model.config.vocab_size
        self.scheduler = Scheduler(
            max_num_seqs, max_model_len, max_num_batched_tokens, self.tokenizer.eos_token_id,
        )
        self.model_runner = ModelRunner(model, max_num_seqs, max_model_len)

    def step(self):
        seqs = self.scheduler.schedule()
        if not seqs:
            return []
        token_ids = self.model_runner.run(seqs)
        return self.scheduler.postprocess(seqs, token_ids)

    def generate(self, prompts, sampling_params=SamplingParams()):
        if not self.scheduler.is_finished():
            raise RuntimeError("finish pending requests before calling generate")
        seqs = [
            Sequence(i, self.tokenizer.encode(p) if isinstance(p, str) else p, sampling_params)
            for i, p in enumerate(prompts)
        ]
        for seq in seqs:
            if not 0 < len(seq.token_ids) < self.scheduler.max_model_len:
                raise ValueError("prompt must be non-empty and leave room for an output token")
            if any(not isinstance(t, int) or not 0 <= t < self.vocab_size for t in seq.token_ids):
                raise ValueError("prompt contains an invalid token ID")
        self.scheduler.waiting.extend(seqs)
        while not self.scheduler.is_finished():
            self.step()
        return [
            {"text": self.tokenizer.decode(s.completion_token_ids, skip_special_tokens=True),
             "token_ids": s.completion_token_ids}
            for s in seqs
        ]
