"""Compare a real checkpoint with Transformers; downloads weights on first use."""

import argparse
import json
import platform

import torch
import transformers

from nano_deepinfer import Engine, SamplingParams
from nano_deepinfer.cache import KVCache
from nano_deepinfer.loading import load_pretrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--revision", default="12fd25f77366fa6b3b4b768ec3050bf629380bac")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tokenizer = load_pretrained(args.model, args.revision)
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False,
    ).eval()
    prompts = ["What is the capital of France?", "Explain a KV cache in one sentence."]
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        for p in prompts
    ]
    max_errors, expected = [], []
    for tokens in inputs:
        ids = torch.tensor([tokens])
        cache = KVCache(model.config, 1, 256)
        slot = cache.allocate()
        # Exercise chunk boundaries, then compare every prompt position.
        chunks = []
        for start in range(0, len(tokens), 7):
            chunk = ids[:, start:start + 7]
            chunks.append(model(chunk, cache, [slot], [start], [chunk.shape[1]]))
        actual = torch.cat(chunks, dim=1)
        with torch.inference_mode():
            target = reference(ids).logits
            torch.testing.assert_close(actual, target, atol=2e-4, rtol=2e-4)
            max_errors.append(float((actual - target).abs().max()))
            generated = reference.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=24,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        expected.append(generated[0, len(tokens):].tolist())
    engine = Engine(model, max_batch_size=2, max_seq_len=256, token_budget=8,
                    eos_token_id=tokenizer.eos_token_id)
    outputs = engine.generate(inputs, SamplingParams(max_tokens=24))
    actual_tokens = [list(output.token_ids) for output in outputs]
    assert actual_tokens == expected, (actual_tokens, expected)
    assert not engine.cache.owned
    print(json.dumps({
        "model": args.model, "revision": args.revision,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "platform": platform.platform(),
        "max_abs_logit_error": max_errors, "greedy_tokens_equal": True,
        "outputs": [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in actual_tokens],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
