"""Dataset download + multivariate LTSF loaders (TSLib-style splits)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset"

# Public mirrors used by Autoformer / TSLib
ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

# Weather / Electricity often hosted on Autoformer google drive; we also accept local drop-in.
DATASET_META = {
    "ETTh1": {"border": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24], "freq": "h"},
    "ETTh2": {"border": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24], "freq": "h"},
    "ETTm1": {
        "border": [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4],
        "freq": "t",
    },
    "ETTm2": {
        "border": [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4],
        "freq": "t",
    },
    "Weather": {"border_ratio": (0.7, 0.1, 0.2), "freq": "t"},
    "Electricity": {"border_ratio": (0.7, 0.1, 0.2), "freq": "h"},
    "Finance": {"border_ratio": (0.7, 0.1, 0.2), "freq": "d"},
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_file(url: str, dest: Path) -> Path:
    import requests

    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def download_ett(names: Optional[List[str]] = None) -> None:
    names = names or list(ETT_URLS.keys())
    for name in names:
        download_file(ETT_URLS[name], DATA_ROOT / "ETT-small" / f"{name}.csv")


def synthesize_weather(n: int = 52696, c: int = 21, seed: int = 0) -> pd.DataFrame:
    """Synthetic Weather-like multivariate series if real CSV is absent."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = {}
    for i in range(c):
        trend = 0.0001 * i * t
        seasonal = 2 * np.sin(2 * np.pi * t / (144 * (1 + i % 5))) + np.cos(
            2 * np.pi * t / (144 * 7)
        )
        noise = rng.normal(0, 0.3, size=n)
        cols[f"V{i}"] = trend + seasonal + noise
    idx = pd.date_range("2020-01-01", periods=n, freq="10min")
    df = pd.DataFrame(cols, index=idx)
    df.insert(0, "date", idx.astype(str))
    return df.reset_index(drop=True)


def synthesize_electricity(n: int = 26304, c: int = 321, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = {}
    for i in range(c):
        daily = np.sin(2 * np.pi * t / 24 + i * 0.01)
        weekly = 0.5 * np.sin(2 * np.pi * t / (24 * 7))
        cols[f"MT_{i:03d}"] = 1.5 + daily + weekly + rng.normal(0, 0.2, size=n)
    idx = pd.date_range("2012-01-01", periods=n, freq="h")
    df = pd.DataFrame(cols, index=idx)
    df.insert(0, "date", idx.astype(str))
    return df.reset_index(drop=True)


def download_finance(
    tickers: Optional[List[str]] = None,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> Path:
    """Download OHLCV and save log-returns multivariate CSV."""
    import yfinance as yf

    tickers = tickers or ["AAPL", "TSLA", "^GSPC"]
    ensure_dir(DATA_ROOT / "finance")
    frames = []
    for tk in tickers:
        raw = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned empty for {tk}")
        # handle MultiIndex columns from newer yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = raw["Close"]
        close = close.astype(float).squeeze()
        logret = np.log(close / close.shift(1))
        vol = logret.rolling(21).std()
        name = tk.replace("^", "")
        frames.append(pd.Series(logret.values, index=logret.index, name=f"{name}_logret"))
        frames.append(pd.Series(vol.values, index=vol.index, name=f"{name}_vol"))
    df = pd.concat(frames, axis=1).dropna()
    df.insert(0, "date", df.index.astype(str))
    out = DATA_ROOT / "finance" / "finance_returns.csv"
    df.reset_index(drop=True).to_csv(out, index=False)
    print(f"Wrote {out} shape={df.shape}")
    return out


def ensure_dataset(name: str) -> Path:
    """Return path to CSV, downloading or synthesizing if needed."""
    if name in ETT_URLS:
        path = DATA_ROOT / "ETT-small" / f"{name}.csv"
        if not path.exists():
            download_ett([name])
        return path
    if name == "Weather":
        path = DATA_ROOT / "weather" / "weather.csv"
        if not path.exists():
            ensure_dir(path.parent)
            # try common Autoformer mirror; fall back to synthetic
            try:
                url = "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather/weather.csv"
                download_file(url, path)
            except Exception as e:
                print(f"Weather download failed ({e}); writing synthetic.")
                synthesize_weather().to_csv(path, index=False)
        return path
    if name == "Electricity":
        path = DATA_ROOT / "electricity" / "electricity.csv"
        if not path.exists():
            ensure_dir(path.parent)
            try:
                url = "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/electricity/electricity.csv"
                download_file(url, path)
            except Exception as e:
                print(f"Electricity download failed ({e}); writing synthetic.")
                # smaller synthetic for 8GB GPU demos
                synthesize_electricity(c=50).to_csv(path, index=False)
        return path
    if name == "Finance":
        path = DATA_ROOT / "finance" / "finance_returns.csv"
        if not path.exists():
            try:
                download_finance()
            except Exception as e:
                print(f"Finance download failed ({e}); writing synthetic returns.")
                ensure_dir(path.parent)
                rng = np.random.default_rng(42)
                n = 2500
                cols = {}
                for name_t in ["AAPL", "TSLA", "GSPC"]:
                    r = rng.normal(0.0002, 0.015, size=n)
                    cols[f"{name_t}_logret"] = r
                    cols[f"{name_t}_vol"] = pd.Series(r).rolling(21).std().bfill().values
                idx = pd.date_range("2015-01-01", periods=n, freq="B")
                df = pd.DataFrame(cols)
                df.insert(0, "date", idx.astype(str))
                df.to_csv(path, index=False)
        return path
    raise ValueError(f"Unknown dataset: {name}")


class Dataset_Forecast(Dataset):
    """Sliding-window multivariate forecasting dataset."""

    def __init__(
        self,
        root_path: Optional[str] = None,
        data_path: Optional[str] = None,
        flag: str = "train",
        size: Optional[List[int]] = None,
        dataset_name: str = "ETTh1",
        scale: bool = True,
        features: str = "M",
        target: str = "OT",
        max_channels: Optional[int] = None,
    ):
        assert flag in ["train", "val", "test"]
        self.seq_len, self.label_len, self.pred_len = size or [96, 48, 96]
        self.flag = flag
        self.scale = scale
        self.features = features
        self.target = target
        self.dataset_name = dataset_name
        self.max_channels = max_channels

        path = Path(data_path) if data_path else ensure_dataset(dataset_name)
        self.root_path = Path(root_path) if root_path else path.parent
        self.data_path = path.name
        self.__read_data__(path)

    def __read_data__(self, path: Path):
        df_raw = pd.read_csv(path)
        # drop date-like cols
        cols = list(df_raw.columns)
        date_cols = [c for c in cols if c.lower() in ("date", "datetime", "time", "timestamp")]
        value_cols = [c for c in cols if c not in date_cols]
        if self.max_channels is not None:
            value_cols = value_cols[: self.max_channels]
        df_data = df_raw[value_cols].astype(float)

        n = len(df_data)
        meta = DATASET_META[self.dataset_name]
        if "border" in meta:
            b1, b2, b3 = meta["border"]
            border1s = [0, b1 - self.seq_len, b2 - self.seq_len]
            border2s = [b1, b2, b3]
        else:
            r0, r1, r2 = meta["border_ratio"]
            n_train = int(n * r0)
            n_val = int(n * r1)
            border1s = [0, n_train - self.seq_len, n_train + n_val - self.seq_len]
            border2s = [n_train, n_train + n_val, n]

        type_map = {"train": 0, "val": 1, "test": 2}
        bid = type_map[self.flag]
        border1, border2 = border1s[bid], border2s[bid]

        self.scaler = StandardScaler()
        train_data = df_data.iloc[border1s[0] : border2s[0]].values
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.n_features = self.data_x.shape[1]

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
        )

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)


def get_dataloader(
    dataset_name: str,
    flag: str,
    seq_len: int,
    pred_len: int,
    batch_size: int = 32,
    max_channels: Optional[int] = None,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
) -> Tuple[DataLoader, Dataset_Forecast]:
    if shuffle is None:
        shuffle = flag == "train"
    ds = Dataset_Forecast(
        dataset_name=dataset_name,
        flag=flag,
        size=[seq_len, seq_len // 2, pred_len],
        max_channels=max_channels,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=flag == "train",
    )
    return loader, ds
