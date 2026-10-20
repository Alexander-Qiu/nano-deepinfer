import pytest
import torch

from nano_deepinfer import Config, Engine, Llama, SamplingParams
from nano_deepinfer.cache import KVCache
from nano_deepinfer.sampling import sample


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return Llama(Config(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, max_position_embeddings=64,
    )).eval()


def evaluate(model, tokens, chunks):
    cache = KVCache(model.config, 1, 64)
    slot = cache.allocate()
    outputs, start = [], 0
    for length in chunks:
        outputs.append(model(
            torch.tensor([tokens[start:start + length]]), cache, [slot], [start], [length],
        ))
        start += length
    return torch.cat(outputs, dim=1)


def test_cached_chunks_match_full_causal_forward(model):
    tokens = [1, 7, 3, 12, 5, 9, 4]
    full = evaluate(model, tokens, [7])
    for chunks in ([3, 2, 1, 1], [1] * 7):
        torch.testing.assert_close(evaluate(model, tokens, chunks), full, atol=2e-6, rtol=2e-5)
    # Future tokens must not influence earlier logits.
    torch.testing.assert_close(evaluate(model, tokens[:3], [3]), full[:, :3])


def test_ragged_batch_matches_individual(model):
    cache = KVCache(model.config, 2, 64)
    slots = [cache.allocate(), cache.allocate()]
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    batched = model(tokens, cache, slots, [0, 0], [4, 2])
    for row, length in enumerate([4, 2]):
        expected = evaluate(model, tokens[row, :length].tolist(), [length])
        torch.testing.assert_close(batched[row, :length], expected[0], atol=2e-6, rtol=2e-5)
    next_logits = model(torch.tensor([[7], [8]]), cache, slots, [4, 2], [1, 1])
    for row, seq in enumerate(([1, 2, 3, 4, 7], [5, 6, 8])):
        expected = evaluate(model, seq, [len(seq)])
        torch.testing.assert_close(next_logits[row, 0], expected[0, -1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_batching_and_chunk_size_preserve_outputs(model, temperature):
    prompts = [[1, 2, 3], [5, 8], [9] * 13, [7, 6]]
    params = SamplingParams(max_tokens=6, temperature=temperature, top_k=10, top_p=0.9, seed=19)
    serial = Engine(model, max_batch_size=1, max_seq_len=64).generate(prompts, params)
    engine = Engine(model, max_batch_size=3, max_seq_len=64, token_budget=3)
    batch = engine.generate(prompts, params)
    assert [o.token_ids for o in serial] == [o.token_ids for o in batch]
    assert not engine.cache.owned and len(engine.cache.free_slots) == 3


def test_late_arrival_budget_and_slot_reuse(model):
    engine = Engine(model, max_batch_size=2, max_seq_len=32, token_budget=3)
    engine.add_request("a", [1, 2, 3, 4, 5], SamplingParams(max_tokens=3))
    assert engine.step() == []
    engine.add_request("b", [2], SamplingParams(max_tokens=1))
    finished = {}
    while engine.has_pending:
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output
        assert sum(item["tokens"] for item in engine.last_step) <= 3
    assert set(finished) == {"a", "b"}
    engine.add_request("c", [6, 7], SamplingParams(max_tokens=3))
    while engine.has_pending:
        output = engine.step()[-1]
    fresh = Engine(model, max_seq_len=32).generate([[6, 7]], SamplingParams(max_tokens=3))[0]
    assert output.token_ids == fresh.token_ids


def test_abort_waiting_and_running(model):
    engine = Engine(model, max_batch_size=1, token_budget=1, max_seq_len=32)
    engine.add_request("a", [1, 2])
    engine.add_request("b", [3])
    engine.step()
    assert engine.abort("b").finish_reason == "aborted"
    assert engine.abort("a").finish_reason == "aborted"
    assert not engine.has_pending and not engine.cache.owned
    assert engine.step() == []
    with pytest.raises(KeyError):
        engine.abort("missing")


def test_eos_and_context_limits(model):
    next_token = int(evaluate(model, [1, 2], [2])[0, -1].argmax())
    eos_engine = Engine(model, max_seq_len=32, eos_token_id=next_token)
    result = eos_engine.generate([[1, 2]])[0]
    assert result.token_ids == (next_token,) and result.finish_reason == "eos"
    result = eos_engine.generate([[1, 2]], SamplingParams(max_tokens=2, ignore_eos=True))[0]
    assert len(result.token_ids) == 2 and result.finish_reason == "length"
    engine = Engine(model, max_seq_len=4, token_budget=1)
    result = engine.generate([[1, 2]], SamplingParams(max_tokens=10))[0]
    assert len(result.token_ids) == 2 and result.finish_reason == "context"
    assert not engine.cache.owned


def test_rejected_requests_do_not_leak(model):
    engine = Engine(model, max_seq_len=8)
    with pytest.raises(ValueError):
        engine.generate([[1], []])
    assert not engine.has_pending
    for prompt in ([], [32], [-1], [1.5], [1] * 8):
        with pytest.raises(ValueError):
            engine.add_request("bad", prompt)
    engine.add_request("a", [1])
    with pytest.raises(ValueError):
        engine.add_request("a", [2])
    with pytest.raises(RuntimeError):
        engine.generate([[3]])
    engine.abort("a")
    with pytest.raises(ValueError):
        Engine(model, token_budget=0)


def test_sampler_filters_and_rng_isolation():
    logits = torch.tensor([1.0, 4.0, 3.0, 2.0])
    rng = torch.Generator().manual_seed(123)
    assert sample(logits, SamplingParams(temperature=0), rng) == 1
    assert sample(logits, SamplingParams(temperature=1e-320), rng) == 1
    assert sample(logits, SamplingParams(temperature=1, top_k=1), rng) == 1
    assert sample(logits, SamplingParams(temperature=1, top_p=0.01), rng) == 1
    params = SamplingParams(temperature=1, top_k=2, top_p=1)
    first, second = (torch.Generator().manual_seed(5) for _ in range(2))
    expected = [sample(logits, params, first) for _ in range(20)]
    torch.rand(100)
    assert expected == [sample(logits, params, second) for _ in range(20)]
    assert set(expected) <= {1, 2}


@pytest.mark.parametrize("kwargs", [
    {"temperature": float("nan")}, {"temperature": -1}, {"top_p": 0},
    {"top_p": 1.1}, {"top_k": -1}, {"max_tokens": 0}, {"seed": -1},
])
def test_bad_sampling_params(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
