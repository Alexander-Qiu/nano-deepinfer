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

代码正在实现和验证中。具体完成范围与验证结果会随代码提交更新。

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
