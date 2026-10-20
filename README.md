# nano-deepinfer

**本仓库完全由 OpenAI Codex 通过 vibe coding 生成，仅供学习与参考。**

**This repository is entirely vibe-coded by OpenAI Codex under human direction, for learning and reference only.**

## 想做什么

参考 [DeepSeek-AI / DeepInfer](https://github.com/deepseek-ai/DeepInfer)，用尽量少、容易读懂的代码，串起大模型推理的基本流程：

```text
请求进入 → 调度 → Prefill / Decode → KV Cache → 采样 → 输出
```

第一版从 **CPU 可运行的教学实现**开始，后续再探索 GPU / NPU。

## 第一版范围

- 纯 PyTorch 的小型 Llama 模型实现。
- 请求调度、分块 Prefill、连续批处理。
- 增量 KV Cache，以及请求结束后的缓存回收。
- Greedy / Temperature / Top-k / Top-p 采样与请求级随机种子。
- 无需下载权重的流程演示，以及真实小模型的加载入口。

首版已实现以上流程，运行环境为 **Python 3.10+、CPU、float32**。GPU / NPU 后端尚未实现。

## 快速运行

```bash
git clone https://github.com/Alexander-Qiu/nano-deepinfer.git
cd nano-deepinfer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# 不下载权重：观察请求调度、KV 位置和输出 token ID
python -m nano_deepinfer --trace --max-tokens 8 --token-budget 8
```

默认演示使用随机权重，输出不具有语言意义。要生成真实文本，安装可选依赖并加载小模型：

```bash
python -m pip install -e '.[hf]'
python -m nano_deepinfer \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --prompt "What is the capital of France?" \
  --max-tokens 24
```

首次运行会下载模型；权重不包含在本仓库中，并遵循模型自身的许可证。已验证输出：`The capital of France is Paris.`

CLI 可重复传入 `--prompt` 来观察多个请求。`--trace` 会打印每一步处理的请求、阶段、KV 槽位和 token 数。

## Python API

```python
from nano_deepinfer import Engine, SamplingParams
from nano_deepinfer.loading import load_pretrained

model, tokenizer = load_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
)
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is the capital of France?"}],
    add_generation_prompt=True,
)
engine = Engine(model, eos_token_id=tokenizer.eos_token_id)
outputs = engine.generate([tokens], SamplingParams(max_tokens=24))
print(tokenizer.decode(outputs[0].token_ids, skip_special_tokens=True))
```

需要逐步观察或动态加入请求时，使用 `engine.add_request()`、`engine.step()`、`engine.abort()`。`step()` 返回本步产生输出的请求及其完整生成 token 序列。

## 验证

```bash
python -m pip install -e '.[hf,test]'
python -m pytest -q
python examples/verify_checkpoint.py
```

本地 CPU 已通过 25 项测试，并完成固定版本 SmolLM2-135M-Instruct 与 Transformers 的数值及 greedy 输出对照。验证条件、误差、实现边界和源码阅读顺序见 [设计与验证说明](docs/design.md)。

## 与 DeepInfer 的关系

这是受 DeepInfer 启发的独立学习项目，并非 DeepSeek 官方项目，也不是 DeepInfer 的精简发行版。

当前目标是理解推理引擎的核心流程，不复现 DeepInfer 的生产规模、性能或完整功能。第一版使用小型 Llama 模型验证流程，**不支持 DeepSeek-V4.1 模型**。

## 使用边界

- 仅供学习、阅读代码和实验参考。
- AI 生成的实现、解释和测试可能存在错误，使用前请自行核实。
- 不应用作生产服务，也不据此推断 DeepInfer 或实际 GPU / NPU 的性能。
- 随机权重演示只验证执行流程，不会生成有意义的语言。

## 致谢

感谢 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 提供公开的推理框架实现，作为本项目的学习参考。
