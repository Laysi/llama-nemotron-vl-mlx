---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/llama-nemotron-rerank-vl-1b-v2
library_name: mlx
pipeline_tag: text-ranking
language:
  - multilingual
tags:
  - mlx
  - apple-silicon
  - reranker
  - cross-encoder
  - multimodal reranking
  - visual-document-retrieval
  - rag
---

# llama-nemotron-rerank-vl-1b-v2-mlx

[`nvidia/llama-nemotron-rerank-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Load it with [`llama-nemotron-vl-mlx`](https://github.com/Laysi/llama-nemotron-vl-mlx).

The model scores how relevant a document page is to a query; the page can be text, an image or both. See NVIDIA's model card for its intended use, training data, evaluation, limitations and ethical considerations. Everything there applies here.

**Built with Llama.**

## What changed from NVIDIA's checkpoint

The weights are NVIDIA's bfloat16 values, unchanged. The conversion (`python -m nemotron_vl_mlx.convert`) changes only how they are stored:

- Parameter names follow the MLX modules.
- The SigLIP patch convolution is stored channels-last.
- The SigLIP attention-pooling head is dropped; the model never uses it.

`config.json` and the tokenizer files are NVIDIA's. The repository has no PyTorch remote code, so `transformers` cannot load it; use NVIDIA's repository for that.

## Use

```bash
uv add "llama-nemotron-vl-mlx @ git+https://github.com/Laysi/llama-nemotron-vl-mlx"
hf download Laysi-ai/llama-nemotron-rerank-vl-1b-v2-mlx --local-dir models/llama-nemotron-rerank-vl-1b-v2-mlx
```

```python
import mlx.core as mx
from PIL import Image
from nemotron_vl_mlx import Passage, Reranker

reranker = Reranker("models/llama-nemotron-rerank-vl-1b-v2-mlx", dtype=mx.float16)
page = Image.open("page-7.png")
logits = reranker(
    "Which sensors does the starter kit include?",
    [Passage(text="The kit ships with ..."), Passage(image=page), Passage(text="Figure 3 ...", image=page)],
)
```

Use float16. Each passage runs at its own length (`padded=True` batches them as the original does).

## Fidelity and speed

These were measured on a Mac mini (M6, 24 GB) with MLX 0.32.3 against PyTorch 2.14.1 on MPS:

- **Preprocessing**: token ids, attention masks and pixel values are identical to NVIDIA's processor.
- **float32** (`MLX_ENABLE_TF32=0`): logits are within 1e-5 of PyTorch float32.
- **float16**: 73 real queries from a RAG service, 50 candidates each (524 with a page image), one image tile. The reference is PyTorch MPS float32.

| | Per query | Max logit difference from the reference | Top-6 set differs | Process footprint |
| --- | --- | --- | --- | --- |
| PyTorch MPS float16 | 7.18 s | 0.0082 | 2 of 73 | 7.3 GB |
| MLX float16 | 4.91 s | 0.0104 | 1 of 73 | 4.0 GB |

## License

- The weights are governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) (`LICENSE`) and the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) (`LICENSE-LLAMA-3.2`, with its `USE_POLICY.md`).
- Attribution notices are in `NOTICE`, and NVIDIA's third-party notices are in `THIRD_PARTY_NOTICES.md`.
- The loading code is Apache 2.0.

Licensed by NVIDIA Corporation under the NVIDIA Open Model License. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
