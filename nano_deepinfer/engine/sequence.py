import torch

from nano_deepinfer.sampling_params import SamplingParams


class Sequence:
    def __init__(self, seq_id: int, token_ids: list[int], params: SamplingParams):
        self.seq_id = seq_id
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.slot = None
        self.params = params
        self.rng = torch.Generator().manual_seed(params.seed)

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]
