import pytest
import torch
from transformers import LlamaConfig

from nano_deepinfer import SamplingParams
from nano_deepinfer.models.llama import Llama
from nano_deepinfer.engine.kv_cache import KVCache
from nano_deepinfer.engine.model_runner import ModelRunner
from nano_deepinfer.engine.scheduler import Scheduler
from nano_deepinfer.engine.sequence import Sequence


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return Llama(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )).eval()


def forward(model, tokens, chunks):
    cache = KVCache(model.config, 1, 64)
    outputs, start = [], 0
    for length in chunks:
        outputs.append(model(torch.tensor([tokens[start:start + length]]),
                             cache, [0], [start], [length]))
        start += length
    return torch.cat(outputs, dim=1)


def generate(model, prompts, params, slots=2, budget=3, context=64, eos=None):
    scheduler = Scheduler(slots, context, budget, eos)
    runner = ModelRunner(model, slots, context)
    seqs = [Sequence(i, p, params) for i, p in enumerate(prompts)]
    scheduler.waiting.extend(seqs)
    while not scheduler.is_finished():
        batch = scheduler.schedule()
        assert batch and len({s.slot for s in batch}) == len(batch)
        assert sum(s.num_scheduled_tokens for s in batch) <= budget
        scheduler.postprocess(batch, runner.run(batch))
    assert sorted(scheduler.free_slots) == list(range(slots))
    return [s.completion_token_ids for s in seqs]


def test_causal_cached_forward(model):
    tokens = [1, 7, 3, 12, 5, 9, 4]
    expected = forward(model, tokens, [7])
    for chunks in ([3, 2, 1, 1], [1] * 7):
        torch.testing.assert_close(forward(model, tokens, chunks), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(forward(model, tokens[:3], [3]), expected[:, :3])


def test_ragged_batch_and_reused_slot(model):
    cache = KVCache(model.config, 2, 64)
    # Old finite KV contents must not affect a shorter request reusing a slot.
    cache.keys.fill_(123)
    cache.values.fill_(-456)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    batched = model(tokens, cache, [0, 1], [0, 0], [4, 2])
    for row, length in enumerate([4, 2]):
        expected = forward(model, tokens[row, :length].tolist(), [length])
        torch.testing.assert_close(batched[row, :length], expected[0], atol=2e-6, rtol=2e-5)
    logits = model(torch.tensor([[7], [8]]), cache, [0, 1], [4, 2], [1, 1])
    for row, seq in enumerate(([1, 2, 3, 4, 7], [5, 6, 8])):
        expected = forward(model, seq, [len(seq)])
        torch.testing.assert_close(logits[row, 0], expected[0, -1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("temperature", [0.0, 0.8, 1e-320])
def test_scheduling_matches_serial_generation(model, temperature):
    prompts = [[1, 2, 3], [5, 8], [9] * 13, [7, 6]]
    params = SamplingParams(max_tokens=6, temperature=temperature, seed=19)
    expected = generate(model, prompts, params, slots=1, budget=64)
    for budget in (1, 3, 64):
        assert generate(model, prompts, params, slots=3, budget=budget) == expected


def test_eos_and_context_limits(model):
    eos = int(forward(model, [1, 2], [2])[0, -1].argmax())
    assert generate(model, [[1, 2]], SamplingParams(), eos=eos) == [[eos]]
    assert len(generate(model, [[1, 2]], SamplingParams(max_tokens=2, ignore_eos=True), eos=eos)[0]) == 2
    assert len(generate(model, [[1, 2]], SamplingParams(max_tokens=10), context=4)[0]) == 2
