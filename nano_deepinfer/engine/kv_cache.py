import torch


class KVCache:
    def __init__(self, config, num_slots, max_seq_len):
        shape = (
            config.num_hidden_layers, num_slots, config.num_key_value_heads,
            max_seq_len, config.hidden_size // config.num_attention_heads,
        )
        self.keys = torch.zeros(shape)
        self.values = torch.zeros(shape)

    def update(self, layer, slots, starts, lengths, key, value):
        for row, (slot, start, length) in enumerate(zip(slots, starts, lengths)):
            self.keys[layer, slot, :, start:start + length] = key[row, :, :length]
            self.values[layer, slot, :, start:start + length] = value[row, :, :length]
        end = max(start + length for start, length in zip(starts, lengths))
        return self.keys[layer, slots, :, :end], self.values[layer, slots, :, :end]
