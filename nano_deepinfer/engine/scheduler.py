from collections import deque


class Scheduler:
    def __init__(self, max_num_seqs, max_model_len, max_num_batched_tokens, eos):
        if min(max_num_seqs, max_model_len, max_num_batched_tokens) < 1:
            raise ValueError("scheduler limits must be positive")
        self.max_model_len = max_model_len
        self.token_budget = max_num_batched_tokens
        self.eos = eos
        self.free_slots = list(reversed(range(max_num_seqs)))
        self.waiting = deque()
        self.running = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def schedule(self):
        scheduled = []
        budget = self.token_budget
        # Prefill first; unfinished chunks stay at the front of the waiting queue.
        while self.waiting and budget:
            seq = self.waiting[0]
            if seq.slot is None:
                if not self.free_slots:
                    break
                seq.slot = self.free_slots.pop()
            seq.num_scheduled_tokens = min(len(seq.token_ids) - seq.num_cached_tokens, budget)
            scheduled.append(seq)
            budget -= seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq.token_ids):
                self.running.append(self.waiting.popleft())
        if scheduled:
            return scheduled
        # Rotate decode requests when the token budget is smaller than the batch.
        for _ in range(min(len(self.running), budget)):
            seq = self.running.popleft()
            seq.num_scheduled_tokens = 1
            scheduled.append(seq)
            self.running.append(seq)
        return scheduled

    def postprocess(self, seqs, token_ids):
        finished = []
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens < len(seq.token_ids):
                continue
            seq.token_ids.append(token_id)
            if (
                (token_id == self.eos and not seq.params.ignore_eos)
                or len(seq.completion_token_ids) == seq.params.max_tokens
                or len(seq.token_ids) == self.max_model_len
            ):
                self.running.remove(seq)
                self.free_slots.append(seq.slot)
                finished.append(seq)
        return finished
