import torch
from torch.nn.utils.rnn import pad_sequence

from .kv_cache import KVCache


class ModelRunner:
    def __init__(self, model, max_num_seqs, max_model_len):
        self.model = model.eval()
        self.cache = KVCache(model.config, max_num_seqs, max_model_len)

    def run(self, seqs):
        starts = [s.num_cached_tokens for s in seqs]
        lengths = [s.num_scheduled_tokens for s in seqs]
        tokens = pad_sequence([
            torch.tensor(s.token_ids[start:start + length])
            for s, start, length in zip(seqs, starts, lengths)
        ], batch_first=True)
        logits = self.model(tokens, self.cache, [s.slot for s in seqs], starts, lengths)
        outputs = []
        for row, (seq, length) in enumerate(zip(seqs, lengths)):
            token = None
            if seq.num_cached_tokens + length == len(seq.token_ids):
                scores = logits[row, length - 1]
                if seq.params.temperature == 0:
                    token = int(scores.argmax())
                else:
                    scores = scores.double() - scores.double().max()
                    probs = (scores / seq.params.temperature).softmax(-1)
                    token = int(torch.multinomial(probs, 1, generator=seq.rng))
            outputs.append(token)
        return outputs
