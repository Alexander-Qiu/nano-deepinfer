# nano-deepinfer

完全由 **OpenAI Codex vibe coding** 生成，仅供学习参考。

参考 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)，架构沿用
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8) 的职责划分：

```text
LLM → Scheduler → Sequence
    → ModelRunner → Llama + KVCache
```

最小 CPU / float32 实现：连续批处理、分块 prefill、增量 KV、greedy / temperature 采样。
使用普通 Llama 小模型，不支持 DeepSeek-V4.1、GPU / NPU。

## 运行

```bash
pip install -e .
python example.py
```

首次运行下载 SmolLM2-135M-Instruct；模型权重遵循其自身许可证。

```python
from nano_deepinfer import LLM, SamplingParams

llm = LLM("HuggingFaceTB/SmolLM2-135M-Instruct")
outputs = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16))
print(outputs[0]["text"])
```

测试：`pip install -e '.[test]' && pytest -q`。包含 KV、调度及 Transformers 数值对照。
