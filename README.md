# llama-nemotron-vl-mlx

An [MLX](https://github.com/ml-explore/mlx) port of NVIDIA's multimodal retrieval models for Apple Silicon:

- [`nvidia/llama-nemotron-rerank-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2): relevance logits for a query and document pages given as text, images or both.
- [`nvidia/llama-nemotron-embed-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2): 2048-dimensional embeddings for queries and for the same kinds of pages.

Both share one architecture: a SigLIP 2 vision encoder, a pixel-shuffle MLP projector and a bidirectional Llama 3.2 1B. The port reproduces NVIDIA's preprocessing without PyTorch and runs the model in MLX. It needs no PyTorch, no `transformers` and no `trust_remote_code`.

**Built with Llama.**

## Install

```bash
uv add "llama-nemotron-vl-mlx @ git+https://github.com/Laysi/llama-nemotron-vl-mlx"
# or: pip install "git+https://github.com/Laysi/llama-nemotron-vl-mlx"
```

The import name is `nemotron_vl_mlx`.

## Weights

The converted weights are on Hugging Face:

- [`Laysi-ai/llama-nemotron-rerank-vl-1b-v2-mlx`](https://huggingface.co/Laysi-ai/llama-nemotron-rerank-vl-1b-v2-mlx)
- [`Laysi-ai/llama-nemotron-embed-vl-1b-v2-mlx`](https://huggingface.co/Laysi-ai/llama-nemotron-embed-vl-1b-v2-mlx)

```bash
hf download Laysi-ai/llama-nemotron-rerank-vl-1b-v2-mlx --local-dir models/llama-nemotron-rerank-vl-1b-v2-mlx
hf download Laysi-ai/llama-nemotron-embed-vl-1b-v2-mlx --local-dir models/llama-nemotron-embed-vl-1b-v2-mlx
```

They hold NVIDIA's bfloat16 values unchanged. Only the parameter names and the patch-convolution layout differ, and the unused SigLIP attention-pooling head is dropped. The loader also reads NVIDIA's original snapshots directly. `python -m nemotron_vl_mlx.convert SRC DST` turns one into the other. `tools/prepare_hub.py` builds the Hugging Face repositories, adding the model cards and license files from `model_cards/`.

## Use

```python
import mlx.core as mx
from PIL import Image
from nemotron_vl_mlx import Embedder, Passage, Reranker

page = Image.open("page-7.png")

reranker = Reranker("models/llama-nemotron-rerank-vl-1b-v2-mlx", dtype=mx.float16)
logits = reranker(
    "Which sensors does the starter kit include?",
    [Passage(text="The kit ships with ..."), Passage(image=page), Passage(text="Figure 3 ...", image=page)],
)  # numpy array, one logit per passage

embedder = Embedder("models/llama-nemotron-embed-vl-1b-v2-mlx", dtype=mx.float16)
query = embedder([Passage(text="Which sensors does the starter kit include?")], "query")
pages = embedder([Passage(image=page), Passage(text="The kit ships with ...")], "passage")
# (n, 2048) float32, not normalised; rank pages by cosine similarity.
```

Notes:

- **Rerank**
  - `Reranker` runs each passage at its own length instead of as one left-padded batch, which skips the padding work. `padded=True` reproduces the original's batching.
  - `max_input_tiles` sets the image tiles per page (default 6, as in NVIDIA's processor). One tile is much faster and was enough for our pages.
  - `max_length` defaults to the tokenizer's 10240. NVIDIA's model card suggests 2048 for image passages, 8192 for text and 10240 for image plus text.
- **Embed**: `Embedder` gives what sentence-transformers gives for the embed model: the `query: ` and `passage: ` prompts, 4096-token truncation, and the mean of the last hidden state.
- **Precision**: use float16. bfloat16 moved rerank logits by up to 0.078 in our tests. MLX runs float32 matrix multiplications as TF32 on M5/M6 GPUs unless `MLX_ENABLE_TF32=0` is set.
- **Memory in long-running processes**: MLX keeps freed buffers for reuse with no practical limit, and passages of different lengths make that cache grow (13.8 GiB after 36 queries in our tests). `mx.set_cache_limit(256 * 2**20)` cost 2% of rerank time.

## Fidelity and speed

These were measured on a Mac mini (M6, 24 GB) with MLX 0.32.3 against PyTorch 2.14.1 on MPS, in Knowva, the on-premise RAG service this port was written for.

**Preprocessing**: token ids, attention masks and pixel values are identical to NVIDIA's processors. The rerank check covers tiles 1 and 6 with text, image, image plus text, mixed batches, truncation and literal `<image>` in a passage (`tools/check_processing.py`). The embed check covers all four input shapes (`tools/check_embed_processing.py`).

**float32**: with `MLX_ENABLE_TF32=0`, rerank logits are within 1e-5 of PyTorch float32, and embed vectors have cosine 1.000000.

**Rerank in float16**: 73 real queries from the RAG service, 50 candidates each, 524 of them with a page image, one tile. The reference is PyTorch MPS float32, which matches PyTorch CPU float32 to 1e-5.

| | Per query | Max logit difference from the reference | Top-6 set differs | Process footprint |
| --- | --- | --- | --- | --- |
| PyTorch MPS float16 | 7.18 s | 0.0082 | 2 of 73 | 7.3 GB |
| MLX float16 | 4.91 s | 0.0104 | 1 of 73 | 4.0 GB |

Where the top six differed from the reference, the reference's 6th and 7th candidates were 0.0018 and 0.0003 apart.

**Embed in float16**: 20 queries and 14 pages, six tiles plus a thumbnail.

| | Query | Text page | Page image | Page image + text | Lowest cosine to PyTorch float32 |
| --- | --- | --- | --- | --- | --- |
| PyTorch MPS float16 | 0.026 s | 0.068 s | 1.00 s | 1.06 s | 0.999992 |
| MLX float16 | 0.022 s | 0.055 s | 0.84 s | 0.91 s | 0.999990 |

Query vectors from MLX retrieve the same top five pages from page vectors that PyTorch computed. A service can therefore switch without re-embedding its documents.

## Checking against PyTorch

```bash
proto install                         # Python and uv from .prototools
uv sync --group reference             # adds NVIDIA's PyTorch implementation
uv run --group reference tools/check_processing.py NVIDIA_RERANK_DIR PAGES_DIR
uv run --group reference tools/check_embed_processing.py NVIDIA_EMBED_DIR PAGES_DIR
uv run --group reference tools/reference.py NVIDIA_RERANK_DIR PAGES_DIR refs/ --device cpu --dtype float32
MLX_ENABLE_TF32=0 uv run tools/compare.py MODEL_DIR refs/ --dtype float32
```

`PAGES_DIR` holds `p-NN.jpg` page images with their text in `p-NN.txt`. `tools/embed_vectors.py` and `tools/compare_embed.py` compare embeddings, and `tools/matmul_peak.py` measures matmul throughput.

## License

- **Code in this repository**: Apache License 2.0 (`LICENSE`, `NOTICE`). It follows NVIDIA's Apache-2.0 model and processing code.
- **Model weights**: governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). The weight repositories on Hugging Face carry both licenses and their notices.

Built with Llama.
