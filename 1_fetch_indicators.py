#!/usr/bin/env python3
"""
1_fetch_indicators.py (high-throughput version) - Extended

Añadidos:
 - Cálculo de macro EMAs sobre 2H (EMA400_2H, EMA500_2H, EMA600_2H).
 - Descarga por batch de datos diarios y cálculo de EMA50_1D / EMA100_1D (opcional).
 - Cálculo de FibSupport_2H (retroceso 61.8% entre swing high/low en lookback).
 - Configuración para controlar lookback fibo, macro_ema_periods_2h, etc.

Usage:
  python 1_fetch_indicators.py
"""
import os
import time
import logging
import math
from typing import Dict, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("yfinance is required. Install with: pip install yfinance") from e

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("fetch_indicators_fast")

# Default config (can be overridden by config.yaml)
DEFAULT_CONFIG = {
    "spreadsheet_id": "",
    "creds_path": "creds/gsheets-service.json",
    "sheets": ["energía", "salud", "commodites", "financiero", "tecnología"],
    "data_provider": "yfinance",
    "intraday_period_days": 60,
    "intraday_interval": "30m",
    "daily_period_days": 800,   # para calcular EMAs largas de 1D si se desea
    "workers": 8,                # for indicator calculation
    "batch_size": 200,          # number of tickers per yfinance bulk download
    "cache_ttl_min": 15,        # cache raw intraday per ticker for X minutes
    "max_download_retries": 3,
    "output_dir": "intermediate",
    "indicators_parquet": "intermediate/indicators.parquet",
    # indicator windows
    "atr_window": 14,
    "rsi_window": 14,
    "mfi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    # ------------ ADDED config options ---------------
    # macro EMAs to compute over 2H (list of spans)
    "macro_ema_periods_2h": [400, 500, 600],   # <-- ADDED
    # compute EMAs on daily timeframe? Set True to batch download daily and compute EMA50_1D/EMA100_1D
    "compute_daily_emas": True,                # <-- ADDED
    "daily_ema_periods": [50, 100],            # <-- ADDED
    # fibo lookback (bars on 2H) and ratio to use (e.g. 0.618)
    "fib_lookback_bars": 200,                  # <-- ADDED
    "fib_ratio": 0.618,                        # <-- ADDED
    # -------------------------------------------------
}

# -----------------------------
# Indicator functions (same as earlier, vectorized)
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd_diff(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def atr(df: pd.DataFrame, window=14) -> pd.Series:
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

def rsi(series: pd.Series, window=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=window-1, adjust=False).mean()
    ma_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ma_up / ma_down
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series

def mfi(df: pd.DataFrame, window=14) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    mf = tp * df['Volume']
    delta_tp = tp.diff()
    pos_mf = mf.where(delta_tp > 0, 0.0)
    neg_mf = mf.where(delta_tp < 0, 0.0)
    pos_sum = pos_mf.rolling(window=window, min_periods=1).sum()
    neg_sum = neg_mf.rolling(window=window, min_periods=1).sum().abs()
    with np.errstate(divide='ignore', invalid='ignore'):
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        mfi_series = 100 - (100 / (1 + mfr))
    mfi_series = mfi_series.fillna(0)
    mfi_series[neg_sum == 0] = 100.0
    return mfi_series

def ema_slope(series_ema: pd.Series, periods_back: int = 5) -> float:
    if series_ema.shape[0] < periods_back + 1:
        return 0.0
    last = series_ema.iloc[-1]
    prev = series_ema.shift(periods_back).iloc[-1]
    if prev == 0 or pd.isna(prev) or pd.isna(last):
        return 0.0
    return float((last - prev) / prev)

# -----------------------------
# Helpers: config, sectors, cache
# -----------------------------
def load_config(path="config.yaml"):
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
            cfg.update(user_cfg)
    os.makedirs(cfg.get("output_dir", "intermediate"), exist_ok=True)
    os.makedirs(os.path.join(cfg.get("output_dir", "intermediate"), "cache"), exist_ok=True)
    return cfg

def read_sectors_csv(path="sectors.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("sectors.csv not found")
    df = pd.read_csv(path, dtype=str)
    cols = set(df.columns.str.strip())
    if not {"Ticker", "Sector"}.issubset(cols):
        raise ValueError("sectors.csv must contain Ticker and Sector columns")
    df = df[['Ticker', 'Sector']].dropna()
    df['Ticker'] = df['Ticker'].str.strip().str.upper()
    df['Sector'] = df['Sector'].str.strip()
    return df

def cache_path_for_ticker(cfg, ticker):
    out = cfg.get("output_dir", "intermediate")
    return os.path.join(out, "cache", f"{ticker}.csv")

def is_cache_fresh(cfg, ticker):
    p = cache_path_for_ticker(cfg, ticker)
    if not os.path.exists(p):
        return False
    mtime = os.path.getmtime(p)
    age_min = (time.time() - mtime) / 60.0
    return age_min <= float(cfg.get("cache_ttl_min", 15))

# -----------------------------
# Bulk download + parsing helpers
# -----------------------------
def bulk_download_intraday(tickers: List[str], cfg: Dict) -> Dict[str, Optional[pd.DataFrame]]:
    intraday_period = f"{int(cfg.get('intraday_period_days', 60))}d"
    interval = cfg.get('intraday_interval', '30m')
    retries = int(cfg.get("max_download_retries", 3))

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            logger.debug("Bulk download attempt %d for %d tickers (interval=%s)", attempt, len(tickers), interval)
            raw = yf.download(tickers, period=intraday_period, interval=interval, progress=False, threads=True, group_by='ticker', auto_adjust=False)
            results = {}
            if isinstance(raw.columns, pd.MultiIndex):
                for t in tickers:
                    if t in raw.columns.levels[0]:
                        df_t = raw[t].dropna(how='all')
                        if not df_t.empty:
                            if df_t.index.tz is None:
                                df_t.index = df_t.index.tz_localize('UTC')
                            else:
                                df_t.index = df_t.index.tz_convert('UTC')
                            results[t] = df_t
                        else:
                            results[t] = None
                    else:
                        results[t] = None
            else:
                if len(tickers) == 1:
                    df_t = raw.copy()
                    if df_t.index.tz is None:
                        df_t.index = df_t.index.tz_localize('UTC')
                    else:
                        df_t.index = df_t.index.tz_convert('UTC')
                    results[tickers[0]] = df_t
                else:
                    for t in tickers:
                        results[t] = None
            return results
        except Exception as e:
            logger.warning("Bulk download attempt %d failed: %s", attempt, e)
            time.sleep(1.0 * attempt)
    logger.error("Bulk download failed after %d attempts for tickers: %s", retries, tickers[:10])
    return {t: None for t in tickers}

def bulk_download_daily(tickers: List[str], cfg: Dict) -> Dict[str, Optional[pd.DataFrame]]:  # <-- ADDED
    """
    Batch download daily OHLCV for tickers. Returns dict ticker->DataFrame (1D).
    """
    daily_period = f"{int(cfg.get('daily_period_days', 365))}d"
    retries = int(cfg.get("max_download_retries", 3))
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            logger.debug("Bulk daily download attempt %d for %d tickers", attempt, len(tickers))
            raw = yf.download(tickers, period=daily_period, interval='1d', progress=False, threads=True, group_by='ticker', auto_adjust=False)
            results = {}
            if isinstance(raw.columns, pd.MultiIndex):
                for t in tickers:
                    if t in raw.columns.levels[0]:
                        df_t = raw[t].dropna(how='all')
                        if not df_t.empty:
                            if df_t.index.tz is None:
                                df_t.index = df_t.index.tz_localize('UTC')
                            else:
                                df_t.index = df_t.index.tz_convert('UTC')
                            results[t] = df_t
                        else:
                            results[t] = None
                    else:
                        results[t] = None
            else:
                if len(tickers) == 1:
                    df_t = raw.copy()
                    if df_t.index.tz is None:
                        df_t.index = df_t.index.tz_localize('UTC')
                    else:
                        df_t.index = df_t.index.tz_convert('UTC')
                    results[tickers[0]] = df_t
                else:
                    for t in tickers:
                        results[t] = None
            return results
        except Exception as e:
            logger.warning("Bulk daily download attempt %d failed: %s", attempt, e)
            time.sleep(1.0 * attempt)
    logger.error("Bulk daily download failed after %d attempts for tickers: %s", retries, tickers[:10])
    return {t: None for t in tickers}

def save_cache(ticker: str, df: pd.DataFrame, cfg: Dict):
    try:
        p = cache_path_for_ticker(cfg, ticker)
        df.to_csv(p)
    except Exception:
        logger.exception("Failed to save cache for %s", ticker)

def load_cache(ticker: str, cfg: Dict) -> Optional[pd.DataFrame]:
    p = cache_path_for_ticker(cfg, ticker)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        return df
    except Exception:
        logger.exception("Failed to load cache for %s", ticker)
        return None

# -----------------------------
# Per-ticker processing (pure function suitable for ProcessPool)
# -----------------------------
def compute_indicators_for_ticker(args):
    """
    args: tuple (ticker, sector, df_2h, df_daily_or_None, cfg)
    This function runs in worker processes (no global state).
    """
    ticker, sector, df_2h, df_daily, cfg = args
    try:
        if df_2h is None or df_2h.empty:
            return None

        # Use last N rows; ensure floats
        df_2h = df_2h[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).dropna(how='any')

        close = df_2h['Close']
        ema15 = ema(close, 15)
        ema50 = ema(close, 50)
        ema150 = ema(close, 150)
        atr14 = atr(df_2h, window=int(cfg.get('atr_window', 14)))
        rsi14 = rsi(close, window=int(cfg.get('rsi_window', 14)))
        mfi14 = mfi(df_2h, window=int(cfg.get('mfi_window', 14)))
        macd_d = macd_diff(close, fast=int(cfg.get('macd_fast', 12)), slow=int(cfg.get('macd_slow', 26)), signal=int(cfg.get('macd_signal', 9)))
        ema150_slope = ema_slope(ema150, periods_back=5)

        # --- ADDED: macro EMAs over 2H (400,500,600 or as in config) ---
        macro_ema_periods = cfg.get("macro_ema_periods_2h", []) or []
        macro_ema_values = {}
        for p in macro_ema_periods:
            try:
                if close.shape[0] >= int(p):
                    macro_ema_values[f"EMA{int(p)}_2H"] = ema(close, int(p)).iloc[-1]
                else:
                    macro_ema_values[f"EMA{int(p)}_2H"] = None
            except Exception:
                macro_ema_values[f"EMA{int(p)}_2H"] = None

        # --- ADDED: daily EMAs if provided (EMA50_1D, EMA100_1D etc.) ---
        daily_ema_values = {}
        if cfg.get("compute_daily_emas", False) and isinstance(df_daily, pd.DataFrame) and not df_daily.empty:
            try:
                df_daily_clean = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).dropna(how='any')
                close_d = df_daily_clean['Close']
                for p in cfg.get("daily_ema_periods", []):
                    try:
                        if close_d.shape[0] >= int(p):
                            daily_ema_values[f"EMA{int(p)}_1D"] = ema(close_d, int(p)).iloc[-1]
                        else:
                            daily_ema_values[f"EMA{int(p)}_1D"] = None
                    except Exception:
                        daily_ema_values[f"EMA{int(p)}_1D"] = None
            except Exception:
                # ignore daily ema errors
                pass

        # --- ADDED: fibonacci support calc (swing high/low over lookback) ---
        fib_support_val = None
        try:
            lookback = int(cfg.get("fib_lookback_bars", 200))
            fib_ratio = float(cfg.get("fib_ratio", 0.618))
            if df_2h.shape[0] >= 10 and lookback > 3:
                use_lb = min(lookback, df_2h.shape[0])
                recent = df_2h.tail(use_lb)
                swing_high = float(recent['High'].max())
                swing_low = float(recent['Low'].min())
                # Fib support = low + (high - low) * fib_ratio
                fib_support_val = swing_low + (swing_high - swing_low) * fib_ratio
        except Exception:
            fib_support_val = None

        def safe_val(s):
            try:
                v = s.iloc[-1]
                if pd.isna(v) or (isinstance(v, float) and math.isinf(v)):
                    return None
                return float(v)
            except Exception:
                return None

        result = {
            "Ticker": ticker,
            "Sector": sector,
            "LastClose": safe_val(close),
            "ATR_2H": safe_val(atr14),
            "EMA15_2H": safe_val(ema15),
            "EMA50_2H": safe_val(ema50),
            "EMA150_2H": safe_val(ema150),
            "EMA150_slope": ema150_slope,
            "MACD_diff": safe_val(macd_d),
            "RSI_2H": safe_val(rsi14),
            "MFI_2H": safe_val(mfi14),
            "Timestamp": str(df_2h.index[-1]) if not df_2h.empty else None,
            # added extras:
            "FibSupport_2H": fib_support_val
        }

        # attach macro EMAs
        for k, v in macro_ema_values.items():
            result[k] = v

        # attach daily EMAs
        for k, v in daily_ema_values.items():
            result[k] = v

        return result
    except Exception as e:
        logger.exception("Error computing indicators for %s: %s", ticker, e)
        return None

# -----------------------------
# Orchestration: batches -> compute -> save
# -----------------------------
def resample_to_2h(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df2h = df.resample('2H').agg(agg).dropna(how='any')
        return df2h
    except Exception:
        logger.exception("Error resampling to 2H")
        return pd.DataFrame()

def main():
    cfg = load_config("config.yaml")
    sectors_df = read_sectors_csv("sectors.csv")
    tickers = sectors_df['Ticker'].tolist()
    sectors_map = dict(zip(sectors_df['Ticker'], sectors_df['Sector']))

    batch_size = int(cfg.get("batch_size", 200))
    workers = max(1, int(cfg.get("workers", 8)))

    logger.info("Starting fetch for %d tickers in batches of %d (workers=%d)", len(tickers), batch_size, workers)

    # Prepare list of batches
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    # collect dataframes per ticker to process indicators
    ticker_dfs = {}       # ticker -> df_2h
    ticker_daily_dfs = {} # <-- ADDED: ticker -> df_daily (or None)

    for batch_num, batch in enumerate(batches, start=1):
        # For each ticker: if cache fresh, load from cache
        to_download = []
        for t in batch:
            if is_cache_fresh(cfg, t):
                df_cached = load_cache(t, cfg)
                if df_cached is not None and not df_cached.empty:
                    ticker_dfs[t] = resample_to_2h(df_cached)
                    continue
            to_download.append(t)

        # Download intraday for to_download
        if to_download:
            logger.info("Batch %d/%d: downloading %d tickers (of %d)", batch_num, len(batches), len(to_download), len(batch))
            downloaded = bulk_download_intraday(to_download, cfg)
            for t in to_download:
                df_t = downloaded.get(t)
                if df_t is not None and not df_t.empty:
                    save_cache(t, df_t, cfg)
                    ticker_dfs[t] = resample_to_2h(df_t)
                else:
                    # attempt fallback: try 1h interval single-ticker fetch
                    logger.debug("Attempting 1h fallback for %s", t)
                    try:
                        df_single = yf.download(t, period=f"{int(cfg.get('intraday_period_days',60))}d", interval='1h', progress=False, threads=False, auto_adjust=False)
                        if df_single is not None and not df_single.empty:
                            if df_single.index.tz is None:
                                df_single.index = df_single.index.tz_localize('UTC')
                            else:
                                df_single.index = df_single.index.tz_convert('UTC')
                            save_cache(t, df_single, cfg)
                            ticker_dfs[t] = resample_to_2h(df_single)
                        else:
                            logger.warning("No data after 1h fallback for %s", t)
                            ticker_dfs[t] = pd.DataFrame()
                    except Exception:
                        logger.exception("Fallback failed for %s", t)
                        ticker_dfs[t] = pd.DataFrame()
        # If compute_daily_emas is True, also batch download daily for this batch
        if cfg.get("compute_daily_emas", False):
            try:
                logger.info("Batch %d/%d: downloading daily data for %d tickers (for daily EMAs)", batch_num, len(batches), len(batch))
                daily_downloaded = bulk_download_daily(batch, cfg)
                for t in batch:
                    df_d = daily_downloaded.get(t)
                    if df_d is not None and not df_d.empty:
                        ticker_daily_dfs[t] = df_d
                    else:
                        # fallback: single-ticker daily fetch
                        try:
                            df_single_d = yf.download(t, period=f"{int(cfg.get('daily_period_days',365))}d", interval='1d', progress=False, threads=False, auto_adjust=False)
                            if df_single_d is not None and not df_single_d.empty:
                                if df_single_d.index.tz is None:
                                    df_single_d.index = df_single_d.index.tz_localize('UTC')
                                else:
                                    df_single_d.index = df_single_d.index.tz_convert('UTC')
                                ticker_daily_dfs[t] = df_single_d
                            else:
                                ticker_daily_dfs[t] = pd.DataFrame()
                        except Exception:
                            ticker_daily_dfs[t] = pd.DataFrame()
            except Exception:
                logger.exception("Daily batch download failed for batch %d", batch_num)
                # ensure keys exist
                for t in batch:
                    if t not in ticker_daily_dfs:
                        ticker_daily_dfs[t] = pd.DataFrame()

    # Prepare args for parallel computation: include df_daily if compute_daily_emas True
    compute_args = []
    for t in tickers:
        df2h = ticker_dfs.get(t)
        df_daily = ticker_daily_dfs.get(t) if cfg.get("compute_daily_emas", False) else None
        compute_args.append((t, sectors_map.get(t, ""), df2h, df_daily, cfg))

    logger.info("Starting indicator calculations in %d workers (processes). Total tickers: %d", workers, len(compute_args))

    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(compute_indicators_for_ticker, arg): arg[0] for arg in compute_args}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                res = fut.result(timeout=300)
                if res is None:
                    failed.append(t)
                else:
                    results.append(res)
            except Exception as e:
                logger.exception("Error in worker for %s: %s", t, e)
                failed.append(t)

    # Build DataFrame and save parquet - ensure new columns exist
    out_df = pd.DataFrame(results)
    # Base desired columns plus newly added ones
    desired_cols = [
        "Ticker", "Sector", "LastClose", "ATR_2H", "EMA15_2H", "EMA50_2H", "EMA150_2H",
        "EMA150_slope", "MACD_diff", "RSI_2H", "MFI_2H", "Timestamp",
        "FibSupport_2H"
    ]
    # Add macro ema columns dynamically from config
    for p in cfg.get("macro_ema_periods_2h", []) or []:
        desired_cols.append(f"EMA{int(p)}_2H")
    # daily EMAs
    if cfg.get("compute_daily_emas", False):
        for p in cfg.get("daily_ema_periods", []) or []:
            desired_cols.append(f"EMA{int(p)}_1D")

    for c in desired_cols:
        if c not in out_df.columns:
            out_df[c] = None
    out_df = out_df[desired_cols]

    out_path = cfg.get("indicators_parquet", "intermediate/indicators.parquet")
    try:
        out_df.to_parquet(out_path, index=False)
        logger.info("Saved indicators parquet: %s (rows=%d). Failed tickers=%d", out_path, out_df.shape[0], len(failed))
    except Exception:
        csv_path = out_path.replace(".parquet", ".csv")
        out_df.to_csv(csv_path, index=False)
        logger.info("Parquet write failed; wrote CSV: %s", csv_path)

    if failed:
        logger.warning("Failed tickers sample: %s", failed[:40])

if __name__ == "__main__":
    main()
