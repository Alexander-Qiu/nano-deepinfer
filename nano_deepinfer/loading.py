"""Optional Hugging Face I/O; execution stays inside our own Llama and Engine."""

import json
from pathlib import Path

from .model import Config, Llama


def load_pretrained(model_id, revision=None):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    path = Path(model_id)
    if not path.is_dir():
        path = Path(snapshot_download(
            model_id, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"],
        ))
    config = Config.from_hf(json.loads((path / "config.json").read_text()))
    model = Llama(config)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        filenames = sorted(set(index["weight_map"].values()))
    else:
        filenames = ["model.safetensors"]
    state = {}
    for filename in filenames:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint shard name must stay inside the model directory")
        # Hub snapshots intentionally symlink shards into a shared blobs directory.
        state.update(load_file(str(path / relative)))
    if config.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    return model.eval(), tokenizer
