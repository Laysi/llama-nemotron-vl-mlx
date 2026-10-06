"""Builds a Hugging Face upload directory from one of NVIDIA's original
snapshots: the converted weights, config and tokenizer, the model card from
model_cards/, NVIDIA's LICENSE and THIRD_PARTY_NOTICES.md, the Llama 3.2
license with its use policy, and the NOTICE both licenses ask for.

    uv run tools/prepare_hub.py rerank|embed NVIDIA_SNAPSHOT_DIR OUT_DIR
"""

import shutil
import sys
from pathlib import Path

from nemotron_vl_mlx.convert import convert

kind, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
if kind not in ("rerank", "embed"):
    sys.exit(__doc__)
cards = Path(__file__).resolve().parent.parent / "model_cards"

convert(src, dst)
shutil.copy2(cards / f"{kind}.md", dst / "README.md")
for name in ("NOTICE", "LICENSE-LLAMA-3.2", "USE_POLICY.md"):
    shutil.copy2(cards / name, dst / name)
for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
    shutil.copy2(src / name, dst / name)
print(f"{dst}: {', '.join(sorted(p.name for p in dst.iterdir()))}")
