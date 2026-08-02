"""Optional adapters note for full THUML Time-Series-Library models.

This repo ships compact faithful PyTorch implementations under `baselines/`
(InformerLite, AutoformerLite, FEDformerLite, PatchTST, iTransformer) so the
project runs without vendoring the full TSLib tree.

To swap in official TSLib checkpoints / modules:

1. Clone https://github.com/thuml/Time-Series-Library into `third_party/Time-Series-Library`
2. Point PYTHONPATH at that repo
3. Wrap their `models/*.py` with the same `(B, L, C) -> (B, H, C)` interface in
   `baselines/tslib_adapters/wrappers.py` (stub below).
"""

from __future__ import annotations

# Stub — extend when TSLib is present
TSLIB_AVAILABLE = False

try:
    import importlib.util
    from pathlib import Path

    tslib = Path(__file__).resolve().parents[2] / "third_party" / "Time-Series-Library"
    if tslib.exists():
        TSLIB_AVAILABLE = True
except Exception:
    TSLIB_AVAILABLE = False


def build_tslib_model(name: str, **kwargs):
    raise NotImplementedError(
        f"Official TSLib adapter for {name} not wired. "
        "Use built-in baselines (Informer/Autoformer/FEDformer/PatchTST/iTransformer) "
        "or install Time-Series-Library under third_party/."
    )
