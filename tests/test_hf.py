"""Independent numerical reference, using tiny random weights without downloads."""

from dataclasses import asdict

import pytest
import torch

transformers = pytest.importorskip("transformers")
from nano_deepinfer import Config, Engine, Llama, SamplingParams
from nano_deepinfer.cache import KVCache
from nano_deepinfer.loading import load_pretrained


@pytest.mark.parametrize("tied", [False, True])
def test_logits_and_greedy_generation_match_transformers(tied):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    config = Config(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, max_position_embeddings=64, tie_word_embeddings=tied,
    )
    hf_config = transformers.LlamaConfig(**asdict(config))
    hf_config._attn_implementation = "eager"
    reference = transformers.LlamaForCausalLM(hf_config).eval()
    model = Llama(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.tensor([[1, 4, 2, 7, 9]])
    cache = KVCache(config, 1, 64)
    slot = cache.allocate()
    actual = model(tokens, cache, [slot], [0], [5])
    with torch.inference_mode():
        expected = reference(tokens).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for _ in range(5):
        with torch.inference_mode():
            token = reference(tokens).logits[:, -1].argmax(-1, keepdim=True)
        tokens = torch.cat((tokens, token), dim=1)
    result = Engine(model, max_seq_len=64, token_budget=2).generate(
        [[1, 4, 2, 7, 9]], SamplingParams(max_tokens=5),
    )[0]
    assert list(result.token_ids) == tokens[0, 5:].tolist()


@pytest.mark.parametrize("symlinked", [False, True])
def test_local_checkpoint_roundtrip(tmp_path, symlinked):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    config = transformers.LlamaConfig(
        vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32, tie_word_embeddings=True,
    )
    reference = transformers.LlamaForCausalLM(config).eval()
    reference.save_pretrained(tmp_path, max_shard_size="1KB")
    backend = Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.save_pretrained(tmp_path)
    if symlinked:
        blobs = tmp_path.parent / (tmp_path.name + "-blobs")
        blobs.mkdir()
        for shard in tmp_path.glob("*.safetensors"):
            target = blobs / shard.name
            shard.rename(target)
            shard.symlink_to(target)
    model, loaded_tokenizer = load_pretrained(tmp_path)
    assert loaded_tokenizer.encode("hello") == [1]
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], tensor)


@pytest.mark.parametrize("patch", [
    {"model_type": "deepseek_v4"}, {"rope_scaling": {"type": "linear", "factor": 2}},
    {"attention_bias": True}, {"head_dim": 128}, {"hidden_act": "gelu"},
])
def test_unsupported_checkpoint_is_rejected(patch):
    data = {"model_type": "llama", **asdict(Config()), **patch}
    with pytest.raises(ValueError):
        Config.from_hf(data)
