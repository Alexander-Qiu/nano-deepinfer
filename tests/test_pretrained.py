import os

import pytest
import torch
from transformers import AutoModelForCausalLM

from nano_deepinfer import LLM, SamplingParams


@pytest.mark.skipif(not os.getenv("CHECKPOINT"), reason="set CHECKPOINT to a local SmolLM2 model")
def test_pretrained_generation():
    torch.set_num_threads(4)
    path = os.environ["CHECKPOINT"]
    llm = LLM(path, max_num_seqs=2, max_num_batched_tokens=8)
    reference = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    prompts = ["What is the capital of France?", "Explain a KV cache in one sentence."]
    inputs = [llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True,
    ) for p in prompts]
    outputs = llm.generate(inputs, SamplingParams(max_tokens=24))
    for tokens, output in zip(inputs, outputs):
        ids = torch.tensor([tokens])
        with torch.inference_mode():
            expected = reference.generate(
                ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=24,
                eos_token_id=llm.tokenizer.eos_token_id, pad_token_id=llm.tokenizer.eos_token_id,
            )
        assert output["token_ids"] == expected[0, len(tokens):].tolist()
