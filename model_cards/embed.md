---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/llama-nemotron-embed-vl-1b-v2
library_name: mlx
pipeline_tag: sentence-similarity
language:
  - multilingual
tags:
  - mlx
  - apple-silicon
  - retrieval
  - visual document retrieval
  - page image embedding
  - text embedding
  - rag
---

# llama-nemotron-embed-vl-1b-v2-mlx

[`nvidia/llama-nemotron-embed-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Load it with [`llama-nemotron-vl-mlx`](https://github.com/Laysi/llama-nemotron-vl-mlx).

The model embeds queries, and document pages given as text, an image or both, into 2048-dimensional vectors for retrieval. See NVIDIA's model card for its intended use, training data, evaluation, limitations and ethical considerations. Everything there applies here.

**Built with Llama.**

## What changed from NVIDIA's checkpoint

The weights are NVIDIA's bfloat16 values, unchanged. The conversion (`python -m nemotron_vl_mlx.convert`) changes only how they are stored:

- Parameter names follow the MLX modules.
- The SigLIP patch convolution is stored channels-last.
- The SigLIP attention-pooling head is dropped; the model never uses it.

`config.json` and the tokenizer files are NVIDIA's. The repository has no PyTorch remote code, so `transformers` and `sentence-transformers` cannot load it; use NVIDIA's repository for those.

## Use

```bash
uv add "llama-nemotron-vl-mlx @ git+https://github.com/Laysi/llama-nemotron-vl-mlx"
hf download Laysi-ai/llama-nemotron-embed-vl-1b-v2-mlx --local-dir models/llama-nemotron-embed-vl-1b-v2-mlx
```

```python
import mlx.core as mx
from PIL import Image
from nemotron_vl_mlx import Embedder, Passage

embedder = Embedder("models/llama-nemotron-embed-vl-1b-v2-mlx", dtype=mx.float16)
query = embedder([Passage(text="Which sensors does the starter kit include?")], "query")
pages = embedder([Passage(image=Image.open("page-7.png")), Passage(text="The kit ships with ...")], "passage")
# (n, 2048) float32, not normalised; rank pages by cosine similarity.
```

The embedder applies what sentence-transformers applies for this model: the `query: ` and `passage: ` prompts, 4096-token truncation, and the mean of the last hidden state. Use float16.

## Fidelity and speed

These were measured on a Mac mini (M6, 24 GB) with MLX 0.32.3 against PyTorch 2.14.1 on MPS: 20 queries and 14 pages, six tiles plus a thumbnail.

- **Preprocessing**: token ids and pixel values are identical to what sentence-transformers feeds the original.
- **float32** (`MLX_ENABLE_TF32=0`): cosine 1.000000 to PyTorch float32 for all four input shapes.
- **float16**:

| | Query | Text page | Page image | Page image + text | Lowest cosine to PyTorch float32 |
| --- | --- | --- | --- | --- | --- |
| PyTorch MPS float16 | 0.026 s | 0.068 s | 1.00 s | 1.06 s | 0.999992 |
| MLX float16 | 0.022 s | 0.055 s | 0.84 s | 0.91 s | 0.999990 |

Query vectors from MLX retrieve the same top five pages from page vectors that PyTorch computed. Documents embedded with the original therefore need no re-embedding.

## License

- The weights are governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) (`LICENSE`) and the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) (`LICENSE-LLAMA-3.2`, with its `USE_POLICY.md`).
- Attribution notices are in `NOTICE`, and NVIDIA's third-party notices are in `THIRD_PARTY_NOTICES.md`.
- The loading code is Apache 2.0.

Licensed by NVIDIA Corporation under the NVIDIA Open Model License. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
