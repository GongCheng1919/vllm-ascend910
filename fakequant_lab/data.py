"""Evaluation / calibration text for the P1 harness.

WikiText-2 is read from the local parquet copy (no network).  The PPL protocol
is the standard one: join the split with newlines, tokenize once, then cut into
non-overlapping `seq_len` windows so every reported token is predicted from a
full context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

WIKITEXT2_DIR = Path("/mnt/local_datasets/eval/wikitext-2-v1/wikitext-2-v1")


def load_wikitext2_text(split: str = "test") -> str:
    import pyarrow.parquet as pq

    matches = sorted(WIKITEXT2_DIR.glob(f"{split}-*.parquet"))
    assert matches, f"no {split} parquet under {WIKITEXT2_DIR}"
    table = pq.read_table(matches[0])
    return "\n\n".join(table.column("text").to_pylist())


def make_windows(text: str, tokenizer, seq_len: int = 2048,
                 max_windows: Optional[int] = None) -> torch.Tensor:
    """Tokenize `text` and cut it into `[nwin, seq_len]` non-overlapping windows."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    nwin = ids.numel() // seq_len
    assert nwin > 0, f"text too short: {ids.numel()} tokens < seq_len={seq_len}"
    if max_windows is not None:
        nwin = min(nwin, max_windows)
    return ids[: nwin * seq_len].reshape(nwin, seq_len).contiguous()


def wikitext2_windows(model_path: str, split: str = "test", seq_len: int = 2048,
                      max_windows: Optional[int] = None) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return make_windows(load_wikitext2_text(split), tok, seq_len, max_windows)


C4_DIR = Path("/mnt/local_datasets/pretrain/c4/en")


def c4_windows(model_path: str, seq_len: int = 2048, n_windows: int = 128,
               seed: int = 0) -> torch.Tensor:
    """Calibration windows from C4, the corpus GPTQ/AWQ conventionally calibrate on.

    Kept separate from the evaluation corpus on purpose: calibrating on
    WikiText-2 train and then scoring WikiText-2 test is in-domain and flatters
    the result, which is the wrong way to set an acceptance threshold.

    Each window is a random 2048-token span of a document long enough to hold
    one, matching the GPTQ paper's protocol.
    """
    import gzip
    import json as _json
    import random

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    files = sorted(C4_DIR.glob("c4-train.*.json.gz"))
    assert files, f"no C4 shards under {C4_DIR}"

    rng = random.Random(seed)
    rows: list = []
    for path in files:
        with gzip.open(path, "rt") as f:
            for line in f:
                ids = tok(_json.loads(line)["text"], return_tensors="pt").input_ids[0]
                if ids.numel() <= seq_len:
                    continue
                s = rng.randint(0, ids.numel() - seq_len - 1)
                rows.append(ids[s : s + seq_len])
                if len(rows) >= n_windows:
                    return torch.stack(rows).contiguous()
    raise RuntimeError(f"C4 exhausted with only {len(rows)}/{n_windows} windows")
