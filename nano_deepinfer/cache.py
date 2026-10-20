"""One dense KV slot per active request; no paging or prefix sharing."""

import torch


class KVCache:
    def __init__(self, config, num_slots: int, max_seq_len: int):
        shape = (
            config.num_hidden_layers, num_slots, config.num_key_value_heads,
            max_seq_len, config.hidden_size // config.num_attention_heads,
        )
        self.keys = torch.zeros(shape)
        self.values = torch.zeros(shape)
        self.max_seq_len = max_seq_len
        self.free_slots = list(reversed(range(num_slots)))
        self.owned: set[int] = set()

    @property
    def nbytes(self) -> int:
        return 2 * self.keys.numel() * self.keys.element_size()

    def allocate(self) -> int:
        if not self.free_slots:
            raise RuntimeError("no free KV slots")
        slot = self.free_slots.pop()
        self.owned.add(slot)
        return slot

    def release(self, slot: int):
        if slot not in self.owned:
            raise ValueError("KV slot is not allocated")
        self.owned.remove(slot)
        # Explicit clearing makes ownership visible and prevents stale padding.
        self.keys[:, slot].zero_()
        self.values[:, slot].zero_()
        self.free_slots.append(slot)

    def update(self, layer, slots, starts, lengths, key, value):
        for row, (slot, start, length) in enumerate(zip(slots, starts, lengths)):
            if slot not in self.owned or start < 0 or start + length > self.max_seq_len:
                raise ValueError("invalid KV write")
            self.keys[layer, slot, :, start:start + length] = key[row, :, :length]
            self.values[layer, slot, :, start:start + length] = value[row, :, :length]
        end = max(start + length for start, length in zip(starts, lengths))
        return self.keys[layer, slots, :, :end], self.values[layer, slots, :, :end]
