import torch

from nano_deepinfer import LLM, SamplingParams


torch.set_num_threads(4)
llm = LLM("HuggingFaceTB/SmolLM2-135M-Instruct",
          revision="12fd25f77366fa6b3b4b768ec3050bf629380bac")
prompts = ["What is the capital of France?", "Explain a KV cache in one sentence."]
inputs = [
    llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
    )
    for prompt in prompts
]
for output in llm.generate(inputs, SamplingParams(max_tokens=24)):
    print(output["text"])
