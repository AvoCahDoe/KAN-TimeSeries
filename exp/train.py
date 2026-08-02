"""Unified train / evaluate harness."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import build_model
from data import get_dataloader
from metrics import compute_metrics, count_parameters


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def result_dir(cfg: Dict[str, Any]) -> Path:
    ds = cfg["data"]["dataset"]
    model = cfg["model"]["name"]
    h = cfg["data"]["pred_len"]
    seed = cfg.get("seed", 2021)
    tag = cfg.get("tag", "")
    base = ROOT / "results" / ds / model / f"h{h}" / f"seed{seed}"
    if tag:
        base = base / tag
    base.mkdir(parents=True, exist_ok=True)
    return base


@torch.no_grad()
def evaluate(model, loader, device, finance: bool = False, capture_aux: bool = False):
    model.eval()
    preds, trues = [], []
    aux_out = None
    for i, (bx, by) in enumerate(loader):
        bx = bx.to(device)
        by = by.to(device)
        if capture_aux and hasattr(model, "forward") and "return_aux" in model.forward.__code__.co_varnames:
            yp, aux_out = model(bx, return_aux=True)
            capture_aux = False  # only first batch
        else:
            yp = model(bx)
        preds.append(yp.cpu().numpy())
        trues.append(by.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    metrics = compute_metrics(pred, true, finance=finance)
    return metrics, pred, true, aux_out


def train_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(cfg.get("seed", 2021))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg.get("train", {})

    seq_len = dcfg["seq_len"]
    pred_len = dcfg["pred_len"]
    batch_size = tcfg.get("batch_size", 32)
    max_channels = dcfg.get("max_channels")
    finance = dcfg.get("dataset") == "Finance" or dcfg.get("finance", False)

    train_loader, train_ds = get_dataloader(
        dcfg["dataset"], "train", seq_len, pred_len, batch_size, max_channels
    )
    val_loader, _ = get_dataloader(
        dcfg["dataset"], "val", seq_len, pred_len, batch_size, max_channels, shuffle=False
    )
    test_loader, _ = get_dataloader(
        dcfg["dataset"], "test", seq_len, pred_len, batch_size, max_channels, shuffle=False
    )
    enc_in = train_ds.n_features

    model_kwargs = dict(mcfg.get("kwargs", {}))
    model_kwargs.update(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        d_model=mcfg.get("d_model", 64),
    )
    # TimeKAN-specific passthrough
    for k in (
        "k_bands",
        "decomp_mode",
        "fusion_mode",
        "n_kan_layers",
        "degree_schedule",
        "uniform_degree",
        "basis",
        "use_revin",
        "dropout",
        "degree",
        "n_layers",
        "individual",
        "patch_len",
        "stride",
        "n_heads",
        "e_layers",
        "modes",
        "moving_avg",
    ):
        if k in mcfg:
            model_kwargs[k] = mcfg[k]
    model = build_model(mcfg["name"], **model_kwargs).to(device)
    n_params = count_parameters(model)

    # Non-trainable models
    trainable = any(p.requires_grad for p in model.parameters())
    out_dir = result_dir(cfg)
    history = {"train_loss": [], "val_mse": []}

    if not trainable or mcfg["name"] in ("Naive", "ARIMA"):
        test_metrics, pred, true, aux = evaluate(
            model, test_loader, device, finance=finance, capture_aux=True
        )
        timing = {"params": n_params, "ms_per_batch": None, "epochs": 0}
        _save_artifacts(out_dir, cfg, test_metrics, pred, true, timing, history, aux=aux)
        return test_metrics

    opt = torch.optim.Adam(
        model.parameters(),
        lr=tcfg.get("lr", 1e-3),
        weight_decay=tcfg.get("weight_decay", 1e-4),
    )
    epochs = tcfg.get("epochs", 10)
    patience = tcfg.get("patience", 3)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            yp = model(bx)
            loss = criterion(yp, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses[-50:])))
        train_loss = float(np.mean(losses))
        val_metrics, _, _, _ = evaluate(model, val_loader, device, finance=finance)
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_metrics["mse"])
        print(
            f"[{mcfg['name']}] epoch={epoch} train={train_loss:.6f} "
            f"val_mse={val_metrics['mse']:.6f}"
        )
        if val_metrics["mse"] < best_val - 1e-8:
            best_val = val_metrics["mse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # latency
    model.eval()
    bx, _ = next(iter(test_loader))
    bx = bx.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_runs = 20
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(bx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_runs * 1000.0

    test_metrics, pred, true, aux = evaluate(
        model, test_loader, device, finance=finance, capture_aux=True
    )
    timing = {
        "params": n_params,
        "ms_per_batch": ms,
        "epochs": len(history["train_loss"]),
        "best_val_mse": best_val,
    }
    _save_artifacts(
        out_dir, cfg, test_metrics, pred, true, timing, history, model=model, aux=aux
    )
    print(f"Test metrics: {test_metrics}")
    print(f"Saved -> {out_dir}")
    return test_metrics


def _save_artifacts(
    out_dir: Path,
    cfg,
    metrics,
    pred,
    true,
    timing,
    history,
    model: Optional[nn.Module] = None,
    aux=None,
):
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    np.savez_compressed(out_dir / "preds.npz", pred=pred, true=true)
    if model is not None:
        torch.save(model.state_dict(), out_dir / "model.pt")
    aux_dict = {}
    if aux is not None:
        if aux.get("attn") is not None:
            a = aux["attn"]
            aux_dict["attn"] = a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        if aux.get("masks") is not None:
            m = aux["masks"]
            aux_dict["masks"] = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        if aux.get("bands") is not None:
            bands = aux["bands"]
            aux_dict["bands"] = np.stack(
                [b.cpu().numpy() if torch.is_tensor(b) else np.asarray(b) for b in bands],
                axis=1,
            )
    elif model is not None and cfg["model"]["name"] == "TimeKAN":
        if getattr(model, "last_attn", None) is not None:
            aux_dict["attn"] = model.last_attn.cpu().numpy()
        if getattr(model, "last_masks", None) is not None:
            m = model.last_masks
            aux_dict["masks"] = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        if getattr(model, "last_bands", None) is not None:
            aux_dict["bands"] = np.stack(
                [b.cpu().numpy() for b in model.last_bands], axis=1
            )
    if aux_dict:
        np.savez_compressed(out_dir / "aux.npz", **aux_dict)


def aggregate_summary(results_root: Path | None = None) -> Path:
    """Walk results/ and write summary.csv."""
    import csv

    results_root = results_root or (ROOT / "results")
    rows = []
    for metrics_path in results_root.rglob("metrics.json"):
        timing_path = metrics_path.parent / "timing.json"
        cfg_path = metrics_path.parent / "config.yaml"
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        timing = {}
        if timing_path.exists():
            with open(timing_path, encoding="utf-8") as f:
                timing = json.load(f)
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        parts = metrics_path.relative_to(results_root).parts
        # dataset / model / hXX / seedY / [tag]
        dataset = parts[0] if len(parts) > 0 else ""
        model = parts[1] if len(parts) > 1 else ""
        horizon = parts[2].lstrip("h") if len(parts) > 2 else ""
        seed = parts[3].lstrip("seed") if len(parts) > 3 else ""
        tag = parts[4] if len(parts) > 5 else (parts[4] if len(parts) > 4 and not parts[4].endswith(".json") else "")
        row = {
            "dataset": dataset,
            "model": model,
            "horizon": int(horizon) if str(horizon).isdigit() else horizon,
            "seed": int(seed) if str(seed).isdigit() else seed,
            "tag": tag,
            **m,
            "params": timing.get("params"),
            "ms_per_batch": timing.get("ms_per_batch"),
        }
        if cfg:
            row["decomp_mode"] = cfg.get("model", {}).get("decomp_mode")
            row["fusion_mode"] = cfg.get("model", {}).get("fusion_mode")
            row["basis"] = cfg.get("model", {}).get("basis")
            row["k_bands"] = cfg.get("model", {}).get("k_bands")
        rows.append(row)

    out = results_root / "summary.csv"
    if not rows:
        out.write_text("dataset,model,horizon,seed\n", encoding="utf-8")
        return out
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_one(cfg)
    if args.aggregate:
        aggregate_summary()


if __name__ == "__main__":
    main()
