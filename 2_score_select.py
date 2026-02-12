#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_score_select_robust_relaxed.py
Versión completa y final del scoring con la lógica MFI/RSI que pediste.
Incluye:
 - Config por defecto (relajada) y lectura de config.yaml.
 - Hard filters relajados con tolerancia y opción para forzar relación MFI<=RSI+diff.
 - Scoring completo con componentes clásicos + nuevos componentes para MFI:
     * mfi_score: pico en 30, decay rápido por debajo de 20 (exponencial)
     * mfi_diff: (RSI - MFI) normalizado y atenuado si MFI < cutoff
 - Fallback para rellenar selección por top global si no se alcanzan candidatos mínimos.
 - Export a parquet/csv y sheet_ready CSV.

Edita parámetros en config.yaml para ajustar comportamiento sin tocar el script.

Uso: python 2_score_select_robust_relaxed.py
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, Tuple, Any

import yaml
import pandas as pd
import numpy as np

# ---- logging ----
LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("score_select_robust_relaxed")

# ---- defaults (más permisivos que el original) ----
DEFAULT_CONFIG: Dict[str, Any] = {
    "indicators_parquet": "intermediate/indicators.parquet",
    "indicators_csv": "intermediate/indicators.csv",
    "candidates_parquet": "intermediate/candidates_relaxed.parquet",
    "candidates_csv": "intermediate/candidates_relaxed.csv",
    # hard filters defaults (relajados)
    "min_atr_distance": 0.35,
    "rsi_2h_max": 70,
    "mfi_2h_max": 60,
    "min_atr": 0.0002,
    "top_n_per_sector": 8,
    "last_n_duplicate_check": 10,
    # macro EMAs hard (col->min_atr_req), vacío por defecto
    "macro_emas_hard": {},
    # fibo soft params
    "use_fib_soft": True,
    "fib_support_col": "FibSupport_2H",
    "fib_ideal_atr": 1.5,
    "fib_max_atr": 2.0,
    # scoring weights (puedes ajustar)
    "weights": {
        "rsi_low": 10,
        "mfi_rsi_soft": 0,   # deprecated by new MFI components, set 0
        "macd_pos": 10,
        "ema15_close_gap": 20,
        "atr_score": 16,
        "ema150_distance": 10,
        "ema50_below": 4,
        "fib_soft": 12,
        # nuevos pesos para MFI
        "mfi_score": 18,
        "mfi_diff": 8
    },
    "ema_gap_atr_threshold": 0.5,
    # NUEVOS parámetros para relajar filtros
    "hard_fail_tolerance": 1,
    "optional_checks": [
        "ema15_above_price",
        "ema50_below_price",
        "price_above_ema150",
        "distance_ok",
        "rsi_ok",
        "mfi_ok"
    ],
    # fallback para asegurar un mínimo de candidatos
    "min_candidates_target": 40,
    "global_fill_limit": 200,
    # parámetros MFI/RSI específicos
    "mfi_rsi_max_diff": 5.0,
    "enforce_mfi_rsi_hard": True,
    # parámetros de forma para scoring MFI
    "mfi_peak": 30.0,
    "mfi_span": 20.0,
    "mfi_cut": 20.0,
    "mfi_decay_k": 0.15,
    "mfi_diff_cap": 30.0,
    "mfi_attenuation_below_cut": 0.7
}

# -------------------------
# Helpers: config + IO
# -------------------------

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
            # deep merge for weights
            w = user.get("weights")
            if w:
                merged = cfg.get("weights", {}).copy()
                merged.update(w)
                cfg["weights"] = merged
            # merge macro_emas_hard if present
            if "macro_emas_hard" in user:
                cfg_mac = cfg.get("macro_emas_hard", {}).copy()
                cfg_mac.update(user.get("macro_emas_hard", {}))
                cfg["macro_emas_hard"] = cfg_mac
            # copy the rest
            for k, v in user.items():
                if k in ("weights", "macro_emas_hard"):
                    continue
                cfg[k] = v
        except Exception as e:
            logger.warning("Error leyendo config.yaml (%s). Usando defaults. Error: %s", path, e)
    else:
        logger.info("config.yaml no encontrado. Usando valores por defecto.")

    # ensure output dir
    outdir = os.path.dirname(cfg.get("candidates_parquet", "intermediate/candidates_relaxed.parquet"))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    return cfg


def read_indicators(cfg: Dict[str, Any]) -> pd.DataFrame:
    p_parquet = cfg.get("indicators_parquet")
    p_csv = cfg.get("indicators_csv")
    df = None
    if p_parquet and os.path.exists(p_parquet):
        try:
            df = pd.read_parquet(p_parquet)
            logger.info("Loaded indicators from parquet: %s (rows=%d)", p_parquet, df.shape[0])
        except Exception as e:
            logger.warning("Could not read parquet %s: %s", p_parquet, e)
            df = None
    if df is None and p_csv and os.path.exists(p_csv):
        try:
            df = pd.read_csv(p_csv)
            logger.info("Loaded indicators from csv: %s (rows=%d)", p_csv, df.shape[0])
        except Exception as e:
            logger.error("Could not read indicators CSV %s: %s", p_csv, e)
            raise
    if df is None:
        raise FileNotFoundError("No indicators file found. Run fetch step first.")
    return df


def safe_float_series(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

# -------------------------
# Safe numeric utils
# -------------------------

def safe_div_series(numer: pd.Series, denom: pd.Series, eps: float = 1e-12) -> pd.Series:
    denom_safe = denom.where(denom > eps, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = numer / denom_safe
    return res

# -------------------------
# Hard filters (relajados / opcionales)
# -------------------------

def apply_hard_filters(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    expected = [
        "Ticker", "Sector", "LastClose", "ATR_2H", "EMA15_2H", "EMA50_2H", "EMA150_2H",
        "RSI_2H", "MFI_2H", "MACD_diff"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # coerce numeric types
    for c in ["LastClose", "ATR_2H", "EMA15_2H", "EMA50_2H", "EMA150_2H", "RSI_2H", "MFI_2H", "MACD_diff"]:
        df[c] = safe_float_series(df[c])

    df["reject_reason"] = ""

    # config
    cfg_min_atr_distance = float(cfg.get("min_atr_distance"))
    cfg_rsi_max = float(cfg.get("rsi_2h_max"))
    cfg_min_atr_abs = float(cfg.get("min_atr"))
    cfg_mfi_max = float(cfg.get("mfi_2h_max"))
    allowed_failures = int(cfg.get("hard_fail_tolerance", 0))
    optional_checks = cfg.get("optional_checks", [])

    # masks
    mask_atr_ok = df["ATR_2H"] >= cfg_min_atr_abs
    mask_atr_ok = mask_atr_ok.fillna(False)

    mask_ema15_above_price = df["EMA15_2H"] > df["LastClose"]
    mask_ema15_above_price = mask_ema15_above_price.fillna(False)

    mask_ema50_below_price = df["EMA50_2H"] < df["LastClose"]
    mask_ema50_below_price = mask_ema50_below_price.fillna(False)

    mask_price_above_ema150 = df["LastClose"] > df["EMA150_2H"]
    mask_price_above_ema150 = mask_price_above_ema150.fillna(False)

    distance_in_atr = safe_div_series(df["LastClose"] - df["EMA150_2H"], df["ATR_2H"], eps=1e-12)
    mask_distance_ok = distance_in_atr >= cfg_min_atr_distance
    mask_distance_ok = mask_distance_ok.fillna(False)

    mask_rsi_ok = df["RSI_2H"] < cfg_rsi_max
    mask_rsi_ok = mask_rsi_ok.fillna(False)

    mask_mfi_ok = df["MFI_2H"] < cfg_mfi_max
    mask_mfi_ok = mask_mfi_ok.fillna(False)

    # Macro EMAs hard check (keep as required if present in config)
    macro_map = cfg.get("macro_emas_hard", {}) or {}
    mask_macro_all = pd.Series(True, index=df.index)
    for col, min_atr_req in macro_map.items():
        if col not in df.columns:
            logger.debug("Macro EMA column '%s' missing in indicators -> ignored", col)
            continue
        dist_macro = safe_div_series(df["LastClose"] - df[col], df["ATR_2H"], eps=1e-12)
        mask_macro_ok = (dist_macro >= float(min_atr_req)).fillna(False)
        mask_macro_all &= mask_macro_ok
        df.loc[~mask_macro_ok, "reject_reason"] = df.loc[~mask_macro_ok, "reject_reason"].astype(str) + (f"; {col}_too_close" if True else "")

    # MFI vs RSI hard/optional check
    mfi_rsi_max_diff = float(cfg.get("mfi_rsi_max_diff", 5.0))
    enforce_mfi_hard = bool(cfg.get("enforce_mfi_rsi_hard", True))
    mask_mfi_vs_rsi = (df["MFI_2H"] <= (df["RSI_2H"] + mfi_rsi_max_diff)).fillna(False)
    df.loc[~mask_mfi_vs_rsi, "reject_reason"] = df.loc[~mask_mfi_vs_rsi, "reject_reason"].astype(str) + ("; mfi_too_high_vs_rsi" if True else "")

    # Build dictionary of masks for optional counting
    mask_dict = {
        "ema15_above_price": mask_ema15_above_price,
        "ema50_below_price": mask_ema50_below_price,
        "price_above_ema150": mask_price_above_ema150,
        "distance_ok": mask_distance_ok,
        "rsi_ok": mask_rsi_ok,
        "mfi_ok": mask_mfi_ok,
        "macro_ok": mask_macro_all,
        "mfi_vs_rsi": mask_mfi_vs_rsi
    }

    # required masks: keep ATR requirement strict to avoid micro-noise
    required_ok = mask_atr_ok.copy()

    # Count optional failures per row
    optional_names = [n for n in optional_checks if n in mask_dict]
    optional_failed = pd.Series(0, index=df.index)
    for name in optional_names:
        optional_failed += (~mask_dict[name]).astype(int)

    # If MFI vs RSI is enforced as hard, include it in required_ok
    if enforce_mfi_hard:
        required_ok &= mask_mfi_vs_rsi

    # Pass if required_ok AND optional_failed <= allowed_failures
    mask_all_relaxed = required_ok & (optional_failed <= allowed_failures)

    # Annotate reject reasons (more informative)
    df.loc[~mask_atr_ok, "reject_reason"] = df.loc[~mask_atr_ok, "reject_reason"].astype(str) + ("; low_atr" if True else "")
    for name in optional_names:
        col_mask = ~mask_dict[name]
        df.loc[col_mask, "reject_reason"] = df.loc[col_mask, "reject_reason"].astype(str) + (f"; {name}_failed" if True else "")

    passed = df[mask_all_relaxed].copy().reset_index(drop=True)
    rejected = df[~mask_all_relaxed].copy().reset_index(drop=True)

    rejected["reject_reason"] = rejected["reject_reason"].astype(str).str.strip("; ").replace({"": None})

    logger.info("Hard filters (relaxed): total=%d passed=%d rejected=%d allowed_failures=%d optional_checks=%s enforce_mfi_hard=%s", df.shape[0], passed.shape[0], rejected.shape[0], allowed_failures, optional_names, enforce_mfi_hard)
    try:
        hist = optional_failed.value_counts().sort_index().to_dict()
        logger.info("Optional failures distribution (fail_count:rows)=%s", hist)
    except Exception:
        pass

    return passed, rejected

# -------------------------
# Scoring (incluye nueva lógica MFI)
# -------------------------

def compute_scores(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    weights = cfg.get("weights", DEFAULT_CONFIG["weights"])

    cols_needed = ["LastClose", "ATR_2H", "EMA15_2H", "EMA50_2H", "EMA150_2H", "RSI_2H", "MFI_2H", "MACD_diff"]
    for c in cols_needed:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = safe_float_series(df[c])

    rsi_max = float(cfg.get("rsi_2h_max", DEFAULT_CONFIG["rsi_2h_max"]))
    rsi_val = df["RSI_2H"].fillna(rsi_max)
    rsi_norm = (rsi_max - rsi_val) / max(rsi_max, 1.0)
    rsi_norm = rsi_norm.clip(lower=0.0, upper=1.0)

    # deprecated soft boolean (kept for compatibility) - we won't use heavily
    mfi_rsi_soft = ((df["MFI_2H"] < df["RSI_2H"]) & (df["MFI_2H"] > 35) & (df["RSI_2H"] > 35)).astype(float).fillna(0.0)

    macd_pos = (df["MACD_diff"] > 0).astype(float).fillna(0.0)

    gap_in_atr = safe_div_series((df["LastClose"] - df["EMA15_2H"]).abs(), df["ATR_2H"], eps=1e-12)
    ema_gap_threshold = float(cfg.get("ema_gap_atr_threshold", 1.0))
    ema_gap_norm = (ema_gap_threshold - gap_in_atr) / max(ema_gap_threshold, 1e-9)
    ema_gap_norm = ema_gap_norm.clip(lower=0.0, upper=1.0).fillna(0.0)

    atr_vals = df["ATR_2H"].fillna(0.0)
    atr_min = atr_vals.min() if not atr_vals.empty else 0.0
    atr_max = atr_vals.max() if not atr_vals.empty else 0.0
    if atr_max - atr_min <= 1e-12:
        atr_norm = pd.Series(0.0, index=df.index)
    else:
        atr_norm = (atr_vals - atr_min) / (atr_max - atr_min)
        atr_norm = atr_norm.clip(0.0, 1.0)

    distance_in_atr = safe_div_series(df["LastClose"] - df["EMA150_2H"], df["ATR_2H"], eps=1e-12)
    cfg_min = float(cfg.get("min_atr_distance", DEFAULT_CONFIG["min_atr_distance"]))
    cfg_max_for_score = max(cfg_min * 4.0, cfg_min + 1.0)
    distance_for_score = distance_in_atr.fillna(cfg_min)
    distance_clipped = (distance_for_score - cfg_min) / max(cfg_max_for_score - cfg_min, 1e-9)
    distance_clipped = distance_clipped.clip(lower=0.0, upper=1.0)

    ema50_below = (df["EMA50_2H"] < df["LastClose"]).astype(float).fillna(0.0)

    # ------------------
    # MFI scoring: triangular peak + exponential decay below cutoff
    mfi_peak = float(cfg.get("mfi_peak", DEFAULT_CONFIG["mfi_peak"]))
    mfi_span = float(cfg.get("mfi_span", DEFAULT_CONFIG["mfi_span"]))
    mfi_cut = float(cfg.get("mfi_cut", DEFAULT_CONFIG["mfi_cut"]))
    decay_k = float(cfg.get("mfi_decay_k", DEFAULT_CONFIG["mfi_decay_k"]))
    diff_cap = float(cfg.get("mfi_diff_cap", DEFAULT_CONFIG["mfi_diff_cap"]))
    attenuation = float(cfg.get("mfi_attenuation_below_cut", DEFAULT_CONFIG["mfi_attenuation_below_cut"]))

    mfi_vals = df["MFI_2H"].fillna(mfi_peak).astype(float)
    rsi_vals = df["RSI_2H"].fillna(rsi_max).astype(float)

    # triangular score centered on mfi_peak; linear drop to 0 at mfi_peak +/- mfi_span
    mfi_tri = 1.0 - (np.abs(mfi_vals - mfi_peak) / max(mfi_span, 1e-9))
    mfi_tri = mfi_tri.clip(lower=0.0, upper=1.0)

    # score at cut point
    raw_score = 1.0 - (np.abs(mfi_cut - mfi_peak) / np.maximum(mfi_span, 1e-9))
    score_at_cut = np.clip(raw_score, 0.0, 1.0)

    mfi_below_mask = mfi_vals < mfi_cut
    # exponential decay below cut (fast decay)
    mfi_below = score_at_cut * np.exp(-decay_k * (mfi_cut - mfi_vals))
    mfi_score = mfi_tri.where(~mfi_below_mask, mfi_below).astype(float).fillna(0.0)

    # distance RSI - MFI (positive is good). cap and normalize
    diff_raw = (rsi_vals - mfi_vals).clip(lower=0.0)
    mfi_diff_score = (diff_raw.clip(upper=diff_cap) / max(diff_cap, 1e-9)).fillna(0.0)
    # attenuate if MFI is very low (avoid overrating extreme low MFI)
    mfi_diff_score = mfi_diff_score * np.where(mfi_vals < mfi_cut, attenuation, 1.0)

    # Fibonacci soft scoring
    fib_soft = pd.Series(0.0, index=df.index)
    if cfg.get("use_fib_soft", True):
        fib_col = cfg.get("fib_support_col", "FibSupport_2H")
        fib_ideal = float(cfg.get("fib_ideal_atr", 1.5))
        fib_max = float(cfg.get("fib_max_atr", 2.0))
        if fib_col in df.columns:
            dist_fib = safe_div_series(df["LastClose"] - df[fib_col], df["ATR_2H"], eps=1e-12).fillna(fib_max * 10.0)
            cond_valid = dist_fib >= 0
            score_vals = 1.0 - (np.abs(dist_fib - fib_ideal) / max(fib_max, 1e-9))
            score_vals = score_vals.clip(lower=0.0, upper=1.0)
            fib_soft = score_vals.where(cond_valid & (dist_fib <= (fib_max * 2.0)), 0.0)
        else:
            logger.debug("Fib column '%s' missing -> fib_soft = 0", fib_col)

    # Compose score components scaled by weights
    score_components: Dict[str, pd.Series] = {}
    score_components["rsi_low"] = rsi_norm * float(weights.get("rsi_low", 0))
    score_components["mfi_rsi_soft"] = mfi_rsi_soft * float(weights.get("mfi_rsi_soft", 0))
    score_components["macd_pos"] = macd_pos * float(weights.get("macd_pos", 0))
    score_components["ema15_close_gap"] = ema_gap_norm * float(weights.get("ema15_close_gap", 0))
    score_components["atr_score"] = atr_norm * float(weights.get("atr_score", 0))
    score_components["ema150_distance"] = distance_clipped * float(weights.get("ema150_distance", 0))
    score_components["ema50_below"] = ema50_below * float(weights.get("ema50_below", 0))
    score_components["fib_soft"] = fib_soft * float(weights.get("fib_soft", 0))

    # nuevos componentes MFI
    score_components["mfi_score"] = pd.Series(mfi_score, index=df.index) * float(weights.get("mfi_score", 0))
    score_components["mfi_diff"] = pd.Series(mfi_diff_score, index=df.index) * float(weights.get("mfi_diff", 0))

    # Sum into total score
    total_score = sum(score_components.values())
    df["Score"] = total_score

    # keep components for debug
    for k, v in score_components.items():
        df[f"_sc_{k}"] = v

    logger.info("Scoring computed. Score stats: min=%.2f median=%.2f max=%.2f", float(df["Score"].min()), float(df["Score"].median()), float(df["Score"].max()))
    return df

# -------------------------
# Selection (top per sector + fallback global)
# -------------------------

def select_top_per_sector(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    top_n = int(cfg.get("top_n_per_sector", 4))
    if "Sector" not in df.columns:
        df["Sector"] = "unknown"
    df_sorted = df.sort_values(by=["Sector", "Score", "ATR_2H", "LastClose"], ascending=[True, False, False, False])
    selected = df_sorted.groupby("Sector", sort=True).head(top_n).reset_index(drop=True)
    logger.info("Selected top %d per sector -> %d rows", top_n, selected.shape[0])
    return selected

# -------------------------
# Prepare sheet-ready export
# -------------------------

def prepare_export_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["TICKER"] = df.get("Ticker", "")
    out["PRECIO"] = df.get("LastClose", "")
    out["ATR 2H"] = df.get("ATR_2H", "")
    out["EMA 15"] = df.get("EMA15_2H", "")
    out["EMA 50"] = df.get("EMA50_2H", "")
    out["EMA 150"] = df.get("EMA150_2H", "")
    out["MFI"] = df.get("MFI_2H", "")
    out["RSI"] = df.get("RSI_2H", "")
    placeholders = ["ENTRADA 1", "ENTRADA 2", "STOP", "TARGET 1", "TARGET 2", "MONTO", "G/P", "G/P POT%"]
    for p in placeholders:
        out[p] = ""
    out["_Score"] = df.get("Score", "")
    out["Sector"] = df.get("Sector", "")
    # include component breakdown for debugging review
    for col in df.columns:
        if col.startswith("_sc_"):
            out[col] = df.get(col, "")
    return out

# -------------------------
# Main
# -------------------------

def main():
    try:
        cfg = load_config("config.yaml")
        df = read_indicators(cfg)

        if df.empty:
            logger.error("Indicators dataframe vacío. Abortando.")
            sys.exit(1)

        passed, rejected = apply_hard_filters(df, cfg)

        out_dir = os.path.dirname(cfg.get("candidates_parquet", "intermediate/candidates_relaxed.parquet")) or "intermediate"
        os.makedirs(out_dir, exist_ok=True)
        out_rejected = os.path.join(out_dir, "rejected_sample_relaxed.csv")
        try:
            rejected.head(500).to_csv(out_rejected, index=False)
            logger.info("Wrote rejected sample to %s", out_rejected)
        except Exception:
            logger.debug("Could not write rejected sample")

        if passed.empty:
            logger.warning("Ningún ticker pasó los filtros (incluso tras relajación). Escribiendo archivos vacíos.")
            empty_df = pd.DataFrame(columns=["Ticker", "Sector"])
            out_parquet = cfg.get("candidates_parquet", "intermediate/candidates_relaxed.parquet")
            out_csv = cfg.get("candidates_csv", "intermediate/candidates_relaxed.csv")
            try:
                empty_df.to_parquet(out_parquet, index=False)
            except Exception:
                empty_df.to_csv(out_csv, index=False)
            sys.exit(0)

        scored = compute_scores(passed, cfg)

        selected = select_top_per_sector(scored, cfg)

        # Fallback: if not enough selected, fill from top global scored
        min_target = int(cfg.get("min_candidates_target", 0))
        if selected.shape[0] < min_target:
            need = max(0, min_target - selected.shape[0])
            logger.info("Selected %d < min_target %d -> rellenando con top global %d", selected.shape[0], min_target, need)
            scored_sorted = scored.sort_values(by=["Score"], ascending=False)
            already = set(selected.get("Ticker", []).astype(str).tolist())
            filler = scored_sorted[~scored_sorted.get("Ticker", "").astype(str).isin(already)].head(min(need, int(cfg.get("global_fill_limit", 200))))
            if not filler.empty:
                selected = pd.concat([selected, filler], ignore_index=True)

        export_df = prepare_export_rows(selected)

        out_parquet = cfg.get("candidates_parquet", "intermediate/candidates_relaxed.parquet")
        out_csv = cfg.get("candidates_csv", "intermediate/candidates_relaxed.csv")
        try:
            selected.to_parquet(out_parquet, index=False)
            logger.info("Wrote selected parquet: %s", out_parquet)
        except Exception:
            selected.to_csv(out_csv, index=False)
            logger.info("Wrote selected csv: %s", out_csv)

        sheet_ready_path = out_parquet.replace(".parquet", ".sheet_ready.csv")
        try:
            export_df.to_csv(sheet_ready_path, index=False)
            logger.info("Wrote sheet-ready CSV: %s", sheet_ready_path)
        except Exception:
            logger.debug("Could not write sheet-ready CSV")

        logger.info("SUMMARY: total_indicators=%d passed=%d selected=%d rejected=%d", df.shape[0], passed.shape[0], selected.shape[0], rejected.shape[0])
        for sector, sub in selected.groupby("Sector"):
            tickers = sub["Ticker"].astype(str).tolist()
            logger.info("Sector=%s -> %s", sector, ", ".join(tickers))

        sys.exit(0)
    except Exception as e:
        logger.exception("Error en scoring relajado: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
