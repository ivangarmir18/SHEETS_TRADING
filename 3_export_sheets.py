#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_export_sheets.py (versión corregida mínima)

- Normaliza nombres (quita tildes y compara en minúsculas)
- Mapea con fuzzy-match entre sheet config y sectores del CSV (fix typo "commodites")
- Busca la primera fila vacía en columna A y escribe en bloque a partir de ahí
- Preserva columnas a la derecha (no borra I..P si sólo escribes A..H)
- Aplica formateo Arial 12 y centrado si gspread_formatting está disponible

Requisitos:
    pip install gspread gspread-formatting pandas pyarrow pyyaml
"""

from __future__ import annotations
import os
import time
import logging
import unicodedata
import difflib
from typing import List, Dict
import yaml
import pandas as pd
import gspread

# Optional formatting
try:
    from gspread_formatting import format_cell_range, cellFormat, textFormat, HorizontalAlignment
    GSFMT = True
except Exception:
    GSFMT = False

# Logging
LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("export_sheets")

# Defaults
DEFAULT_CONFIG = {
    "creds_path": "creds/gsheets-service.json",
    "spreadsheet_id": "",
    "sheets": ["energía", "salud", "commodites", "financiero", "tecnología"],
    "candidates_parquet": "intermediate/candidates.parquet",
    "candidates_csv": "intermediate/candidates.csv",
    "candidates_sheetready_csv": "intermediate/candidates.sheet_ready.csv",
    "top_n_per_sector": 4,
    "last_n_duplicate_check": 6,
    "append_value_input_option": "USER_ENTERED",
    "append_retries": 3,
    "append_retry_delay_sec": 1,
    "fuzzy_cutoff": 0.6,
}

EXPECTED_HEADER = ["TICKER","PRECIO","ATR 2H","EMA 15","EMA 50","EMA 150","MFI","RSI",
                   "ENTRADA 1","ENTRADA 2","STOP","TARGET 1","TARGET 2","MONTO","G/P","G/P POT%"]

# ---------------------------
# Utilities
# ---------------------------
def script_dir():
    return os.path.dirname(os.path.realpath(__file__))

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def load_config(path="config.yaml") -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    p = path if os.path.exists(path) else os.path.join(script_dir(), path)
    if not os.path.exists(p):
        logger.warning("config.yaml no encontrado en %s ni en %s: usando defaults", path, os.path.join(script_dir(), path))
        return cfg
    with open(p, "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
        cfg.update(user)
    return cfg

def resolve_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    alt = os.path.join(script_dir(), p)
    return p if os.path.exists(p) else alt

def read_candidates(cfg: Dict) -> pd.DataFrame:
    candidates_paths = [
        resolve_path(cfg.get("candidates_sheetready_csv", "")),
        resolve_path(cfg.get("candidates_parquet", "")),
        resolve_path(cfg.get("candidates_csv", "")),
    ]
    for p in candidates_paths:
        if p and os.path.exists(p):
            logger.info("Leyendo candidates desde: %s", p)
            if p.lower().endswith(".csv"):
                return pd.read_csv(p, dtype=str).fillna("")
            if p.lower().endswith(".parquet"):
                return pd.read_parquet(p).astype(str).fillna("")
    raise FileNotFoundError("No se encontró candidates.sheet_ready.csv / parquet / csv en paths: " + str(candidates_paths))

def gspread_auth(creds_path: str):
    p = resolve_path(creds_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Service account JSON no encontrado en {creds_path} ni en {p}")
    logger.info("Autenticando gspread con %s", p)
    return gspread.service_account(filename=p)

def find_worksheet_by_normalized_or_exact(spreadsheet, wanted_name: str):
    want = normalize_text(wanted_name)
    for ws in spreadsheet.worksheets():
        if normalize_text(ws.title) == want:
            return ws
    # try exact title (case-sensitive) last resort
    try:
        return spreadsheet.worksheet(wanted_name)
    except Exception:
        return None

def fuzzy_map_sheet_to_sector_name(sheet_name: str, sector_values: List[str], cutoff: float=0.6):
    want = normalize_text(sheet_name)
    unique_norm = {normalize_text(v): v for v in sector_values}
    candidates = list(unique_norm.keys())
    matches = difflib.get_close_matches(want, candidates, n=1, cutoff=cutoff)
    if matches:
        matched_norm = matches[0]
        return unique_norm[matched_norm]
    return None

def format_number_to_sheet_str(value, decimals=2):
    if value is None:
        return ""
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        f = float(s.replace(",", "."))
        fmt = ("{0:." + str(decimals) + "f}").format(f)
        return fmt.replace(".", ",")
    except Exception:
        return s.replace(".", ",")

def apply_formatting(ws, header_len, start_row_idx, end_row_idx):
    if not GSFMT:
        return
    try:
        hdr_end = gspread.utils.rowcol_to_a1(1, header_len)
        fmt_header = cellFormat(textFormat=textFormat(bold=True, fontFamily="Arial", fontSize=12),
                                horizontalAlignment=HorizontalAlignment.CENTER)
        format_cell_range(ws, f"A1:{hdr_end}", fmt_header)
        rows_range = f"{gspread.utils.rowcol_to_a1(start_row_idx,1)}:{gspread.utils.rowcol_to_a1(end_row_idx, header_len)}"
        fmt_rows = cellFormat(textFormat=textFormat(fontFamily="Arial", fontSize=11),
                              horizontalAlignment=HorizontalAlignment.CENTER)
        format_cell_range(ws, rows_range, fmt_rows)
    except Exception as e:
        logger.warning("Formateo falló: %s", e)

def colnum_to_letter(n):
    # 1 -> A, 27 -> AA
    string = ""
    while n > 0:
        n, rem = divmod(n-1, 26)
        string = chr(65 + rem) + string
    return string

# ---------------------------
# Main export logic
# ---------------------------
def main():
    cfg = load_config("config.yaml")
    logger.info("CONFIG: creds_path=%s spreadsheet_id=%s candidates=%s sheets=%s",
                cfg.get("creds_path"), cfg.get("spreadsheet_id"), cfg.get("candidates_sheetready_csv"), cfg.get("sheets"))

    # Read candidates
    try:
        df = read_candidates(cfg)
    except Exception as e:
        logger.exception("No se pudo leer candidates: %s", e)
        return

    # Ensure TICKER and Sector column exist
    cols_lower = [c.strip().lower() for c in df.columns]
    if "ticker" not in cols_lower:
        logger.error("El CSV debe tener columna 'TICKER'. Columnas encontradas: %s", df.columns.tolist())
        return
    sector_candidates = [c for c in df.columns if c.strip().lower() in ("sector", "_sector")]
    if not sector_candidates:
        logger.error("No se encontró columna 'Sector' o '_Sector' en el CSV. Columnas: %s", df.columns.tolist())
        return
    sector_col = sector_candidates[0]
    logger.info("Usando columna de sector: %s", sector_col)

    # Unique sectors for fuzzy mapping
    unique_sectors = sorted(df[sector_col].dropna().unique().tolist())

    # Authenticate gspread
    try:
        gc = gspread_auth(cfg.get("creds_path"))
        sh = gc.open_by_key(cfg.get("spreadsheet_id"))
    except Exception as e:
        logger.exception("Error al autenticar/abrir spreadsheet: %s", e)
        return

    logger.info("Spreadsheet abierto: %s", sh.title)

    top_n = int(cfg.get("top_n_per_sector", 4))
    last_n = int(cfg.get("last_n_duplicate_check", 6))
    fuzzy_cutoff = float(cfg.get("fuzzy_cutoff", 0.6))

    # For each sheet in config
    for sheet_cfg in cfg.get("sheets", []):
        logger.info("Procesando hoja config: '%s'", sheet_cfg)
        ws = find_worksheet_by_normalized_or_exact(sh, sheet_cfg)

        if ws is None:
            logger.info("Hoja '%s' no encontrada exactamente: se creará (title exacto = config)", sheet_cfg)
            try:
                ws = sh.add_worksheet(title=sheet_cfg, rows="1000", cols=str(len(EXPECTED_HEADER)))
                ws.insert_row(EXPECTED_HEADER, index=1)
                logger.info("Hoja creada y cabecera escrita: %s", sheet_cfg)
            except Exception as e:
                logger.exception("No pude crear hoja '%s': %s", sheet_cfg, e)
                continue

        # Map sector: exact normalized match first
        want_norm = normalize_text(sheet_cfg)
        sector_values_list = df[sector_col].astype(str).tolist()
        mask_exact = [normalize_text(sv) == want_norm for sv in sector_values_list]
        df_sector = df[mask_exact]
        mapped_sector_used = None
        if df_sector.empty:
            mapped_sector = fuzzy_map_sheet_to_sector_name(sheet_cfg, unique_sectors, cutoff=fuzzy_cutoff)
            if mapped_sector:
                logger.info("Sheet '%s' mapeada por fuzzy a sector CSV '%s'", sheet_cfg, mapped_sector)
                mapped_sector_used = mapped_sector
                mask_fuzzy = [normalize_text(sv) == normalize_text(mapped_sector) for sv in sector_values_list]
                df_sector = df[mask_fuzzy]
            else:
                logger.info("No se encontraron candidatos para sheet '%s' (ni exact ni fuzzy).", sheet_cfg)

        if df_sector.empty:
            logger.info("No hay candidatos para sheet '%s'. Saltando.", sheet_cfg)
            continue

        top_df = df_sector.head(top_n)

        # Ensure header
        values_all = ws.get_all_values()
        header_row = values_all[0] if values_all else []
        if not header_row or len(header_row) < len(EXPECTED_HEADER):
            try:
                ws.insert_row(EXPECTED_HEADER, index=1)
                header_row = EXPECTED_HEADER
                logger.info("Header insertado/actualizado en hoja '%s'.", ws.title)
            except Exception as e:
                logger.warning("No se pudo insertar header en '%s': %s", ws.title, e)

        # existing last tickers from column A
        colA = ws.col_values(1)
        data_colA = colA[1:] if colA and normalize_text(colA[0]) == normalize_text(EXPECTED_HEADER[0]) else colA
        existing_last_tickers = [c for c in data_colA if str(c).strip() != ""][-last_n:]
        logger.debug("Recientes en hoja '%s' (últimos %d): %s", ws.title, last_n, existing_last_tickers)

        # Build rows_to_write preserving EXPECTED_HEADER order
        rows_to_write = []
        tickers_to_write = []
        for _, r in top_df.iterrows():
            ticker = str(r.get("TICKER", "")).strip()
            if not ticker:
                continue
            if ticker in existing_last_tickers:
                logger.info("Ticker %s omitido en '%s' por duplicado reciente.", ticker, ws.title)
                continue
            out = []
            for colname in EXPECTED_HEADER:
                val = r.get(colname, "")
                if colname in ("PRECIO","ATR 2H","EMA 15","EMA 50","EMA 150","MFI","RSI","MONTO","G/P","G/P POT%"):
                    out.append(format_number_to_sheet_str(val, decimals=2))
                else:
                    out.append("" if pd.isna(val) else str(val))
            rows_to_write.append(out)
            tickers_to_write.append(ticker)

        if not rows_to_write:
            logger.info("No hay filas nuevas para añadir en '%s' (tras filtrar duplicados).", ws.title)
            continue

        # Find first empty row in column A (start search after header if present)
        full_colA = ws.col_values(1)
        header_present = (len(full_colA) >= 1 and normalize_text(full_colA[0]) == normalize_text(EXPECTED_HEADER[0]))
        start_search_idx = 1 if header_present else 0
        first_empty_row = None
        for idx in range(start_search_idx, len(full_colA)):
            val = full_colA[idx] if idx < len(full_colA) else ""
            if str(val).strip() == "":
                first_empty_row = idx + 1  # 1-based
                break
        if first_empty_row is None:
            first_empty_row = len(full_colA) + 1

        write_start = first_empty_row

        # ---------------------
        # Escritura robusta que preserva columnas a la derecha
        # ---------------------
        # 1) calcular la columna máxima con algún valor en las filas a escribir
        last_nonempty_per_row = []
        for r in rows_to_write:
            last_idx = -1
            for j, v in enumerate(r):
                if str(v).strip() != "":
                    last_idx = j
            last_nonempty_per_row.append(last_idx)
        global_last = max(last_nonempty_per_row)
        if global_last == -1:
            logger.info("Todas las filas a escribir están vacías -> nada que hacer en '%s'.", ws.title)
            continue
        write_cols = global_last + 1  # número de columnas a escribir

        # 2) leer filas existentes para fusionar
        existing_values = ws.get_all_values()
        first_idx = write_start - 1
        existing_slice = []
        for i in range(len(rows_to_write)):
            idx = first_idx + i
            if idx < len(existing_values):
                existing_slice.append(existing_values[idx])
            else:
                existing_slice.append([])

        # 3) construir merged_rows
        merged_rows = []
        for i, new_row in enumerate(rows_to_write):
            row_last = last_nonempty_per_row[i]
            existing_row = existing_slice[i] if i < len(existing_slice) else []
            if len(existing_row) < write_cols:
                existing_row = existing_row + [""] * (write_cols - len(existing_row))
            merged = list(existing_row[:write_cols])
            for j in range(0, row_last + 1):
                merged[j] = new_row[j]
            merged_rows.append(merged)

        # 4) actualizar bloque (A{write_start}:{end_col}{end_row})
        end_col_letter = colnum_to_letter(write_cols)
        end_row_idx = write_start + len(merged_rows) - 1
        range_to_update = f"A{write_start}:{end_col_letter}{end_row_idx}"

        try:
            logger.info("Escribiendo bloque en '%s' rango %s (cols=%d) filas=%d (tickers: %s)",
                        ws.title, range_to_update, write_cols, len(merged_rows), ", ".join(tickers_to_write))
            ws.update(range_to_update, merged_rows, value_input_option=cfg.get("append_value_input_option", "USER_ENTERED"))
            # Formateo: header (row1) y filas nuevas
            apply_formatting(ws, len(EXPECTED_HEADER), 1, 1)
            apply_formatting(ws, len(EXPECTED_HEADER), write_start, end_row_idx)
            logger.info("Hoja '%s': escritas %d filas correctamente (preservando columnas derecha).", ws.title, len(merged_rows))
        except Exception as e:
            logger.exception("Fallo al escribir bloque en hoja '%s' rango %s: %s", ws.title, range_to_update, e)
            continue

    logger.info("Export finalizado.")

if __name__ == "__main__":
    main()
