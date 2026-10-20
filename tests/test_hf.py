import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from nano_deepinfer import LLM, SamplingParams
from nano_deepinfer.models.llama import Llama
from nano_deepinfer.engine.kv_cache import KVCache


@pytest.mark.parametrize("tied", [False, True])
def test_checkpoint_loading_logits_and_generation(tmp_path, tied):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    config = LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=tied,
    )
    config._attn_implementation = "eager"
    reference = LlamaForCausalLM(config).eval()
    reference.save_pretrained(tmp_path, max_shard_size="1KB")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(tmp_path)
    # Hub snapshots symlink shards outside the snapshot directory.
    blobs = tmp_path.parent / (tmp_path.name + "-blobs")
    blobs.mkdir()
    for shard in tmp_path.glob("*.safetensors"):
        target = blobs / shard.name
        shard.rename(target)
        shard.symlink_to(target)
    llm = LLM(tmp_path, max_model_len=64, max_num_batched_tokens=2)
    model = llm.model_runner.model
    tokens = torch.tensor([[1, 4, 2, 7, 9]])
    actual = model(tokens, KVCache(config, 1, 64), [0], [0], [5])
    with torch.inference_mode():
        torch.testing.assert_close(actual, reference(tokens).logits, atol=2e-6, rtol=2e-5)
        for _ in range(5):
            token = reference(tokens).logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, token), dim=1)
    result = llm.generate([[1, 4, 2, 7, 9]], SamplingParams(max_tokens=5))[0]
    assert result["token_ids"] == tokens[0, 5:].tolist()
    assert llm.generate(["hello"], SamplingParams(max_tokens=1))
    assert llm.generate([]) == []
    for prompt in ([], [32], [-1], [1.5], [1] * 64):
        with pytest.raises(ValueError):
            llm.generate([[1], prompt])
        assert llm.scheduler.is_finished()


def test_reject_unsupported_model():
    with pytest.raises(ValueError):
        Llama(LlamaConfig(attention_bias=True))
