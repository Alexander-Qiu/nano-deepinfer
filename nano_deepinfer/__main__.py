import argparse
import json
import time

import torch

from . import Config, Engine, Llama, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="CPU-first educational inference")
    parser.add_argument("--model", help="Hugging Face Llama checkpoint or local directory")
    parser.add_argument("--revision", help="Checkpoint revision; use a commit for reproducibility")
    parser.add_argument("--prompt", action="append", help="Repeat for multiple requests")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    prompts = args.prompt or ["Explain what a KV cache does in one sentence.", "The capital of France is"]
    tokenizer = None
    if args.model:
        from .loading import load_pretrained
        model, tokenizer = load_pretrained(args.model, args.revision)
        if tokenizer.chat_template:
            inputs = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], add_generation_prompt=True
                ) for p in prompts
            ]
        else:
            inputs = [tokenizer.encode(p) for p in prompts]
        eos = tokenizer.eos_token_id
    else:
        print("Random-weight demo: output token IDs are NOT meaningful text.")
        model = Llama(Config(max_position_embeddings=args.max_seq_len))
        inputs = [list(p.encode("utf-8")) for p in prompts]
        eos = None
    engine = Engine(
        model, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        token_budget=args.token_budget, eos_token_id=eos,
    )
    params = SamplingParams(
        max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed,
    )
    for i, tokens in enumerate(inputs):
        engine.add_request(str(i), tokens, params)
    print(f"KV allocation: {engine.cache.nbytes / 2**20:.2f} MiB (CPU float32)")
    started = time.perf_counter()
    completed, steps = {}, 0
    while engine.has_pending:
        outputs = engine.step()
        steps += 1
        if args.trace:
            print(json.dumps({"step": steps, "batch": engine.last_step}))
        for output in outputs:
            if output.finished:
                completed[output.request_id] = output
    elapsed = time.perf_counter() - started
    for i, prompt in enumerate(prompts):
        output = completed[str(i)]
        text = tokenizer.decode(output.token_ids, skip_special_tokens=True) if tokenizer else list(output.token_ids)
        print(f"\n[{i}] {prompt}\n{text}\nfinish={output.finish_reason}")
    count = sum(len(output.token_ids) for output in completed.values())
    print(f"\n{count} output tokens, {steps} steps, {elapsed:.3f}s, {count / elapsed:.1f} output tokens/s")
    print("Timing includes prefill and scheduling; excludes model loading. Not a decode-only benchmark.")


if __name__ == "__main__":
    main()
