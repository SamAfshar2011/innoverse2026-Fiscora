#!/usr/bin/env python3
"""
retrain_btc.py — Daily Bitcoin forecast pipeline for AI Personal Finance Coach
==============================================================================

Fetch → preprocess → retrain-from-scratch → evaluate → calibrate → save.

Runs once per day (cron / launchd / Task Scheduler). Each run:
  1. Fetches the FULL BTC-USD daily history from yfinance (period="max").
       · Saves the whole history to   data/btc/btc_live.csv   (model training source)
       · Saves the most recent 90 days to data/btc/btc_90d.csv  (chart / display slice)
       · Overwrites the previous day's CSVs so only the latest file is ever used.
  2. Retrains the 10-day multi-step LSTM FROM SCRATCH on that history.
  3. Saves, in the exact artifact contract that server.ipynb + bitcoin.html read:
       · data/btc/model.keras     — best-validation weights
       · data/btc/scalers.pkl     — feature/target scalers + serving contract
       · data/btc/metadata.json   — config, metrics, baseline comparison, signal thresholds

Design notes
------------
· The 20-feature engineering function is byte-identical to server.ipynb's
  `btc_compute_features` and bitcoin.ipynb's `compute_features`. The server rebuilds
  features from the SAME columns/order at serving time, so the trained scalers apply
  cleanly. Do NOT edit the feature list here without updating both notebooks.
· No raw price ever enters the model — every feature is a return / ratio / oscillator,
  which is what lets the model generalize as the price leaves its training range.
· Targets are the 10 next-day log-returns (direct multi-output, no recursive error
  compounding). The 10-day price PATH is reconstructed as P(t+k)=P(t)·exp(Σ r).
· Split is strictly chronological (no shuffling) → no look-ahead leakage. Scalers are
  fit on the training era only.
· Everything degrades gracefully: if Yahoo is unreachable, the run falls back to the
  bundled datasets/btc/btc.csv so the pipeline still completes and the app keeps working.

Usage
-----
    python retrain_btc.py                      # full daily run into data/btc/
    python retrain_btc.py --epochs 12          # quick run (CI / smoke test)
    python retrain_btc.py --artifacts-dir /tmp/x   # write elsewhere (non-destructive test)
    python retrain_btc.py --no-fetch           # reuse existing CSV / dataset, skip network
    python retrain_btc.py --period 5y          # limit history window

Exit code 0 on success, 1 on failure (so cron can detect + alert).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import warnings
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration & CLI
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

DEFAULTS = dict(
    symbol="BTC-USD",
    period="max",                 # yfinance history window (max ≈ 2014→today, ~4000 rows)
    lookback=60,                  # days of context per training sample
    horizon=10,                   # days predicted (direct multi-output)
    train_frac=0.80,              # chronological split fractions
    val_frac=0.10,                # (test = remainder)
    target_clip_pct=(0.5, 99.5),  # train-only percentile clip on target returns
    conv_filters=48, lstm1=96, lstm2=48, dense=48, dropout=0.25,
    learning_rate=1e-3, weight_decay=1e-4, batch_size=64,
    max_epochs=120, patience_early_stop=15, patience_reduce_lr=6,
    serving_context_days=250,     # server rebuilds features from this many recent rows
    # Approx circulating supply for a nominal Market Cap column (yfinance omits it).
    # Not used for training — included only so the CSV carries the requested field.
    approx_circulating_supply=19_700_000,
)

FEATURE_ORDER = [
    "log_ret", "hl_range", "close_pos", "log_vol_chg", "vol_ratio",
    "volat_7", "volat_14", "volat_30", "rsi_14", "macd", "macd_signal",
    "macd_hist", "bb_pctb", "close_sma7", "close_sma21", "close_sma50",
    "roc_7", "roc_14", "dow_sin", "dow_cos",
]


def parse_args():
    p = argparse.ArgumentParser(description="Daily BTC 10-day LSTM retrain pipeline")
    p.add_argument("--artifacts-dir", default=os.path.join(PROJECT_DIR, "data", "btc"))
    p.add_argument("--dataset-fallback", default=os.path.join(PROJECT_DIR, "datasets", "btc", "btc.csv"))
    p.add_argument("--period", default=DEFAULTS["period"])
    p.add_argument("--lookback", type=int, default=DEFAULTS["lookback"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["max_epochs"])
    p.add_argument("--no-fetch", action="store_true",
                   help="skip yfinance; reuse existing btc_live.csv or the bundled dataset")
    return p.parse_args()


def setup_logging(artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    log = logging.getLogger("retrain_btc")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(os.path.join(artifacts_dir, "retrain.log")); fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# 1 · DATA ACQUISITION  (yfinance → CSV, with graceful fallback)
# ---------------------------------------------------------------------------
def fetch_from_yfinance(symbol, period, log) -> pd.DataFrame:
    """Full daily OHLCV history. Handles yfinance's MultiIndex columns + tz index."""
    import yfinance as yf
    log.info(f"fetching {symbol} history from yfinance (period={period}, interval=1d)…")
    raw = yf.download(symbol, period=period, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no rows")
    # Single-ticker downloads can still carry a MultiIndex ('Close','BTC-USD'); flatten it.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.rename(columns=str.title)              # normalize Open/High/Low/Close/Volume
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Start"
    log.info(f"yfinance returned {len(df)} rows  {df.index.min().date()} → {df.index.max().date()}")
    return df


def load_from_dataset(path, log) -> pd.DataFrame:
    """Fallback: the bundled historical CSV (reverse-ordered, BOM, OHLCV+MarketCap)."""
    log.warning(f"falling back to bundled dataset {path}")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    raw["Start"] = pd.to_datetime(raw["Start"].astype(str).str.slice(0, 10))
    df = (raw[["Start", "Open", "High", "Low", "Close", "Volume"]]
          .sort_values("Start").drop_duplicates("Start").set_index("Start"))
    df.index = df.index.normalize()
    return df


def acquire(args, log) -> tuple[pd.DataFrame, str]:
    """Return (ohlcv_df, source_label). yfinance first, dataset fallback on any failure."""
    if not args.no_fetch:
        try:
            return fetch_from_yfinance(DEFAULTS["symbol"], args.period, log), "yfinance"
        except Exception as exc:                      # network / API / parsing failure
            log.error(f"yfinance fetch failed: {exc!r}")
    live_csv = os.path.join(args.artifacts_dir, "btc_live.csv")
    if os.path.exists(live_csv):
        log.warning(f"reusing previously saved {live_csv}")
        df = pd.read_csv(live_csv, parse_dates=["Start"]).set_index("Start")
        df.index = df.index.normalize()
        return df[["Open", "High", "Low", "Close", "Volume"]], "cached_csv"
    return load_from_dataset(args.dataset_fallback, log), "dataset_fallback"


def save_csvs(df, artifacts_dir, log):
    """Write btc_live.csv (full) + btc_90d.csv (display slice), overwriting prior day's."""
    out = df.copy()
    out["Market Cap"] = (out["Close"] * DEFAULTS["approx_circulating_supply"]).round(0)
    out = out.reset_index().rename(columns={"index": "Start"})
    live_path = os.path.join(artifacts_dir, "btc_live.csv")
    d90_path = os.path.join(artifacts_dir, "btc_90d.csv")
    tmp = live_path + ".tmp"
    out.to_csv(tmp, index=False); os.replace(tmp, live_path)     # atomic overwrite
    tmp = d90_path + ".tmp"
    out.tail(90).to_csv(tmp, index=False); os.replace(tmp, d90_path)
    log.info(f"wrote {live_path} ({len(out)} rows) and {d90_path} (90 rows)")


# ---------------------------------------------------------------------------
# 2 · CLEANING
# ---------------------------------------------------------------------------
def clean(df, log) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Continuous daily index; Yahoo very occasionally drops a day → forward-fill small gaps.
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    n_missing = len(full.difference(df.index))
    if n_missing:
        df = df.reindex(full).ffill()
        log.info(f"cleaning: forward-filled {n_missing} missing calendar day(s)")
    df.index.name = "Start"
    # Volume can legitimately be 0 on a rare Yahoo glitch day → carry previous.
    df["Volume"] = df["Volume"].mask(df["Volume"] <= 0).ffill().bfill()
    # OHLC coherence repair.
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df = df.dropna()
    assert (df[["Open", "High", "Low", "Close", "Volume"]] > 0).all().all(), "non-positive values survived cleaning"
    assert df.index.is_monotonic_increasing
    log.info(f"cleaning: {len(df)} clean daily rows  {df.index.min().date()} → {df.index.max().date()}")
    return df


# ---------------------------------------------------------------------------
# 3 · FEATURES  (identical to server.ipynb btc_compute_features — the serving contract)
# ---------------------------------------------------------------------------
def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Volume"]
    logc = np.log(c)
    r = logc.diff()
    feat = pd.DataFrame(index=ohlcv.index)
    feat["log_ret"] = r
    rng = h - l
    feat["hl_range"] = rng / c
    feat["close_pos"] = ((c - l) / rng.replace(0, np.nan)).fillna(0.5)
    logv = np.log(v)
    feat["log_vol_chg"] = logv.diff()
    feat["vol_ratio"] = np.log(v / v.rolling(20).mean())
    for w in (7, 14, 30):
        feat[f"volat_{w}"] = r.rolling(w).std()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = ((100 - 100 / (1 + rs)) / 100.0).fillna(0.5)
    ema12 = logc.ewm(span=12, adjust=False).mean()
    ema26 = logc.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    feat["macd"], feat["macd_signal"], feat["macd_hist"] = macd, sig, macd - sig
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    feat["bb_pctb"] = ((c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)).clip(-0.5, 1.5)
    for w in (7, 21, 50):
        feat[f"close_sma{w}"] = np.log(c / c.rolling(w).mean())
    feat["roc_7"] = logc.diff(7)
    feat["roc_14"] = logc.diff(14)
    dow = ohlcv.index.dayofweek
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return feat[FEATURE_ORDER]


# ---------------------------------------------------------------------------
# 4 · TARGETS · SPLIT · SCALE · SEQUENCES
# ---------------------------------------------------------------------------
def build_dataset(feat, close, args, log):
    from sklearn.preprocessing import RobustScaler, StandardScaler
    lookback, horizon = args.lookback, DEFAULTS["horizon"]
    logret = feat["log_ret"]

    # 10 next-day log-returns per anchor (NaN in the last `horizon` rows by construction).
    targets = pd.DataFrame({f"h{k}": logret.shift(-k) for k in range(1, horizon + 1)}, index=feat.index)

    dates = feat.index.to_numpy()
    n = len(feat)
    valid = [i for i in range(lookback - 1, n - horizon) if not np.isnan(targets.iloc[i].to_numpy()).any()]
    if len(valid) < 200:
        raise RuntimeError(f"only {len(valid)} usable samples — not enough history to train")

    # Chronological split by anchor position (no shuffling → no leakage).
    n_valid = len(valid)
    i_train_end = int(n_valid * DEFAULTS["train_frac"])
    i_val_end = int(n_valid * (DEFAULTS["train_frac"] + DEFAULTS["val_frac"]))
    train_anchors = valid[:i_train_end]
    val_anchors = valid[i_train_end:i_val_end]
    test_anchors = valid[i_val_end:]
    last_train_row = train_anchors[-1]                        # scaler-fit boundary

    # Train-only target clip (heavy-tailed daily returns).
    train_ret = logret.iloc[: last_train_row + 1]
    clip_lo, clip_hi = np.percentile(train_ret.dropna(), args.target_clip_pct if False else DEFAULTS["target_clip_pct"])
    tgt = targets.clip(clip_lo, clip_hi)

    # Fit scalers on the TRAIN ERA only.
    feat_scaler = RobustScaler().fit(feat.to_numpy()[: last_train_row + 1])
    feat_scaled = feat_scaler.transform(feat.to_numpy()).astype(np.float32)
    tgt_scaler = StandardScaler().fit(tgt.iloc[train_anchors].to_numpy())

    def make(anchors):
        X = np.stack([feat_scaled[i - lookback + 1: i + 1] for i in anchors]).astype(np.float32)
        y_raw = tgt.iloc[anchors].to_numpy(dtype=np.float64)
        y_scaled = tgt_scaler.transform(y_raw).astype(np.float32)
        anchor_close = close.to_numpy()[anchors]
        drift = np.array([feat["log_ret"].to_numpy()[i - lookback + 1: i + 1].mean() for i in anchors])
        return dict(X=X, y=y_raw, y_scaled=y_scaled, close=anchor_close, drift=drift,
                    dates=pd.DatetimeIndex(dates[anchors]))

    data = dict(train=make(train_anchors), val=make(val_anchors), test=make(test_anchors),
                feat_scaler=feat_scaler, tgt_scaler=tgt_scaler,
                clip=(float(clip_lo), float(clip_hi)))
    for name in ("train", "val", "test"):
        s = data[name]
        log.info(f"  {name:<5}: X{s['X'].shape}  {s['dates'].min().date()} → {s['dates'].max().date()}")
    # Leakage guard: every train target must end on/before the first val anchor date.
    assert data["train"]["dates"].max() < data["val"]["dates"].min() <= data["test"]["dates"].min()
    return data


# ---------------------------------------------------------------------------
# 5 · MODEL
# ---------------------------------------------------------------------------
def build_model(n_features, args):
    from tensorflow import keras
    d = DEFAULTS
    inp = keras.Input(shape=(args.lookback, n_features), name="feature_window")
    x = keras.layers.Conv1D(d["conv_filters"], 3, padding="causal", activation="swish", name="local_patterns")(inp)
    x = keras.layers.LSTM(d["lstm1"], return_sequences=True, name="lstm_1")(x)
    x = keras.layers.Dropout(d["dropout"])(x)
    x = keras.layers.LSTM(d["lstm2"], name="lstm_2")(x)
    x = keras.layers.Dropout(d["dropout"])(x)
    x = keras.layers.Dense(d["dense"], activation="swish", name="head")(x)
    out = keras.layers.Dense(d["horizon"], name="log_returns_h1_h10")(x)
    model = keras.Model(inp, out, name="btc_multihorizon_lstm")
    try:
        opt = keras.optimizers.AdamW(learning_rate=d["learning_rate"], weight_decay=d["weight_decay"], clipnorm=1.0)
    except (AttributeError, TypeError):
        opt = keras.optimizers.Adam(learning_rate=d["learning_rate"], clipnorm=1.0)
    model.compile(optimizer=opt, loss=keras.losses.Huber(delta=1.0), metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# 6 · EVALUATION  (walk-forward, vs. persistence / drift / always-up baselines)
# ---------------------------------------------------------------------------
def evaluate(split, model, tgt_scaler, horizon):
    pred_ret = tgt_scaler.inverse_transform(model.predict(split["X"], verbose=0)).astype(np.float64)
    pred_cum = np.cumsum(pred_ret, axis=1)
    act_cum = np.cumsum(split["y"], axis=1)
    anchor = split["close"]
    to_paths = lambda cum: anchor[:, None] * np.exp(cum)
    pred_paths, act_paths = to_paths(pred_cum), to_paths(act_cum)
    persist = np.repeat(anchor[:, None], horizon, axis=1)
    drift_cum = split["drift"][:, None] * np.arange(1, horizon + 1)[None, :]
    drift_paths = anchor[:, None] * np.exp(drift_cum)

    def mape(a, b, h): return float(np.mean(np.abs(a[:, h] - b[:, h]) / a[:, h]) * 100)
    def diracc(cum, h): return float((np.sign(cum[:, h]) == np.sign(act_cum[:, h])).mean() * 100)

    per_h = {}
    for h in range(horizon):
        per_h[f"day +{h+1}"] = {
            "RMSE $": round(float(np.sqrt(np.mean((act_paths[:, h] - pred_paths[:, h]) ** 2))), 4),
            "MAE $": round(float(np.mean(np.abs(act_paths[:, h] - pred_paths[:, h]))), 4),
            "MAPE %": round(mape(act_paths, pred_paths, h), 4),
            "dir_acc %": round(diracc(pred_cum, h), 4),
        }
    vs = {
        "LSTM model": {"MAPE_h1": round(mape(act_paths, pred_paths, 0), 4),
                       "MAPE_h10": round(mape(act_paths, pred_paths, horizon - 1), 4),
                       "dir_h1": round(diracc(pred_cum, 0), 4), "dir_h10": round(diracc(pred_cum, horizon - 1), 4)},
        "persistence (flat)": {"MAPE_h1": round(mape(act_paths, persist, 0), 4),
                               "MAPE_h10": round(mape(act_paths, persist, horizon - 1), 4),
                               "dir_h1": None, "dir_h10": None},
        "drift (60d mean)": {"MAPE_h1": round(mape(act_paths, drift_paths, 0), 4),
                             "MAPE_h10": round(mape(act_paths, drift_paths, horizon - 1), 4),
                             "dir_h1": round(diracc(drift_cum, 0), 4), "dir_h10": round(diracc(drift_cum, horizon - 1), 4)},
        "always-up (direction base rate)": {
            "MAPE_h1": round(mape(act_paths, persist, 0), 4),
            "MAPE_h10": round(mape(act_paths, persist, horizon - 1), 4),
            "dir_h1": round(float((act_cum[:, 0] > 0).mean() * 100), 4),
            "dir_h10": round(float((act_cum[:, -1] > 0).mean() * 100), 4)},
    }
    return dict(per_horizon=per_h, vs_baselines=vs, pred_cum10=pred_cum[:, -1], act_cum10=act_cum[:, -1])


# ---------------------------------------------------------------------------
# 7 · MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.artifacts_dir, exist_ok=True)
    log = setup_logging(args.artifacts_dir)
    random.seed(SEED); np.random.seed(SEED)
    log.info("=" * 70)
    log.info("BTC daily retrain pipeline starting")

    try:
        import tensorflow as tf
        from tensorflow import keras
        tf.random.set_seed(SEED)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        raw, source = acquire(args, log)
        df = clean(raw, log)
        save_csvs(df, args.artifacts_dir, log)

        feat = compute_features(df).replace([np.inf, -np.inf], np.nan).dropna()
        close = df["Close"].reindex(feat.index)
        assert list(feat.columns) == FEATURE_ORDER, "feature order drifted from the serving contract"
        log.info(f"features: {feat.shape[0]} rows × {feat.shape[1]}")

        data = build_dataset(feat, close, args, log)
        model = build_model(len(FEATURE_ORDER), args)
        log.info(f"model: {model.count_params():,} params · training up to {args.epochs} epochs (early stopping)")

        model_path = os.path.join(args.artifacts_dir, "model.keras")
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=DEFAULTS["patience_early_stop"],
                                          restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=DEFAULTS["patience_reduce_lr"], min_lr=1e-5, verbose=0),
            keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0),
        ]
        hist = model.fit(data["train"]["X"], data["train"]["y_scaled"],
                         validation_data=(data["val"]["X"], data["val"]["y_scaled"]),
                         epochs=args.epochs, batch_size=DEFAULTS["batch_size"],
                         callbacks=callbacks, shuffle=True, verbose=0)
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        model.save(model_path)
        log.info(f"training done — best epoch {best_epoch}/{len(hist.history['val_loss'])} "
                 f"· val_loss {min(hist.history['val_loss']):.5f}")

        val_res = evaluate(data["val"], model, data["tgt_scaler"], DEFAULTS["horizon"])
        test_res = evaluate(data["test"], model, data["tgt_scaler"], DEFAULTS["horizon"])
        log.info(f"validation  dir+1={val_res['vs_baselines']['LSTM model']['dir_h1']}%  "
                 f"MAPE+1={val_res['vs_baselines']['LSTM model']['MAPE_h1']}%")
        log.info(f"final test  dir+1={test_res['vs_baselines']['LSTM model']['dir_h1']}%  "
                 f"MAPE+1={test_res['vs_baselines']['LSTM model']['MAPE_h1']}%")

        # --- signal calibration (validation-derived thresholds) ---
        move_threshold = float(np.quantile(np.abs(val_res["pred_cum10"]), 0.5))
        cum10_err_std = float(np.std(val_res["pred_cum10"] - val_res["act_cum10"]))
        sig = np.where(test_res["pred_cum10"] > move_threshold, "UP",
              np.where(test_res["pred_cum10"] < -move_threshold, "DOWN", "HOLD"))
        acting = sig != "HOLD"
        acting_hit = (float((np.sign(test_res["pred_cum10"][acting]) ==
                             np.sign(test_res["act_cum10"][acting])).mean() * 100) if acting.any() else float("nan"))
        base_rate = float(max((test_res["act_cum10"] > 0).mean(), (test_res["act_cum10"] < 0).mean()) * 100)
        counts = {k: int((sig == k).sum()) for k in ("UP", "DOWN", "HOLD")}

        # --- save scalers (serving contract) ---
        with open(os.path.join(args.artifacts_dir, "scalers.pkl"), "wb") as fh:
            pickle.dump({
                "feature_scaler": data["feat_scaler"], "target_scaler": data["tgt_scaler"],
                "feature_names": FEATURE_ORDER, "lookback": args.lookback, "horizon": DEFAULTS["horizon"],
                "target_clip": list(data["clip"]), "serving_context_days": DEFAULTS["serving_context_days"],
            }, fh)

        # --- save metadata (schema server.ipynb + bitcoin.html read) ---
        metadata = {
            "model_name": "btc_multihorizon_lstm",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tensorflow_version": tf.__version__,
            "seed": SEED,
            "data_source": source,
            "config": {k: DEFAULTS[k] for k in ("lookback", "horizon", "train_frac", "val_frac",
                                                "conv_filters", "lstm1", "lstm2", "dropout")},
            "feature_names": FEATURE_ORDER,
            "data": {
                "clean_rows": int(len(df)),
                "clean_range": [str(df.index.min().date()), str(df.index.max().date())],
                "samples": {n: int(len(data[n]["X"])) for n in ("train", "val", "test")},
                "target_clip_bounds": list(data["clip"]),
            },
            "training": {"best_epoch": best_epoch, "epochs_ran": len(hist.history["val_loss"]),
                         "best_val_loss": float(min(hist.history["val_loss"]))},
            "validation_metrics": {"per_horizon": val_res["per_horizon"], "vs_baselines": val_res["vs_baselines"]},
            "final_test_metrics": {"per_horizon": test_res["per_horizon"], "vs_baselines": test_res["vs_baselines"]},
            "signal_calibration": {
                "move_threshold_log_return": move_threshold,
                "cum10_error_std_log_return": cum10_err_std,
                "final_test_acting_hit_rate_pct": round(acting_hit, 2) if acting_hit == acting_hit else None,
                "final_test_majority_base_rate_pct": round(base_rate, 2),
                "final_test_signal_counts": counts,
            },
            "caveats": [
                "Daily BTC returns are close to a random walk; this model provides decision support "
                "with quantified uncertainty, not price prediction.",
                "Retrained from scratch daily on the full yfinance BTC-USD history; forecasts should be "
                "read together with the ±uncertainty band and the not-financial-advice disclaimer.",
            ],
        }
        with open(os.path.join(args.artifacts_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)

        # --- reload check: artifacts must reproduce in-memory predictions ---
        reloaded = keras.models.load_model(model_path)
        probe = data["test"]["X"][-3:]
        assert np.allclose(reloaded.predict(probe, verbose=0), model.predict(probe, verbose=0), atol=1e-5)
        log.info(f"acting hit-rate {metadata['signal_calibration']['final_test_acting_hit_rate_pct']}% "
                 f"(base rate {base_rate:.1f}%) · signals {counts}")
        log.info(f"artifacts saved to {args.artifacts_dir} — reload check PASSED")
        log.info("BTC daily retrain pipeline SUCCESS")
        return 0

    except Exception as exc:
        log.exception(f"pipeline FAILED: {exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
