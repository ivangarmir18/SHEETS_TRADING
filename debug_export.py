#!/usr/bin/env python3
# debug_export_normalized.py  -- dry-run diagnóstico (normaliza acentos y mayúsculas)

import os, yaml, pandas as pd, unicodedata
import gspread

def normalize(s):
    if s is None:
        return ""
    s = str(s).strip()
    # NFD, eliminar marcas diacríticas (tildes), pasar a lower
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise SystemExit(f"config.yaml no encontrado en {os.getcwd()}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg

def find_worksheet_by_normalized_name(sh, wanted_name):
    want_n = normalize(wanted_name)
    for ws in sh.worksheets():
        if normalize(ws.title) == want_n:
            return ws
    return None

def main():
    cfg = load_config("config.yaml")
    creds_path = cfg.get("creds_path", "creds/gsheets-service.json")
    ss_id = cfg.get("spreadsheet_id", "")
    sheets = cfg.get("sheets", [])
    top_n = int(cfg.get("top_n_per_sector", 4))
    last_n = int(cfg.get("last_n_duplicate_check", 6))
    candidates_path = cfg.get("candidates_sheetready_csv", "intermediate/candidates.sheet_ready.csv")

    print("CONFIG:")
    print(" creds_path:", creds_path)
    print(" spreadsheet_id:", ss_id)
    print(" candidates csv:", candidates_path)
    print(" sheets:", sheets)
    print(" top_n_per_sector:", top_n, " last_n_duplicate_check:", last_n)
    print("="*60)

    if not os.path.exists(creds_path):
        print("ERROR: Credenciales no encontradas en:", creds_path); return
    if not os.path.exists(candidates_path):
        print("ERROR: candidates.sheet_ready.csv no encontrado en:", candidates_path); return

    df = pd.read_csv(candidates_path, dtype=str).fillna("")
    # detecta columna de sector (Sector o _Sector)
    sector_cols = [c for c in df.columns if c.strip().lower() in ("sector","_sector")]
    if not sector_cols:
        print("ERROR: no se encontró columna 'Sector' en CSV. Columnas:", df.columns.tolist()); return
    sector_col = sector_cols[0]

    try:
        gc = gspread.service_account(filename=creds_path)
        sh = gc.open_by_key(ss_id)
    except Exception as e:
        print("ERROR: fallo al autenticar/abrir spreadsheet:", repr(e)); return
    print("Spreadsheet abierto:", sh.title)
    print()

    for sheet_name in sheets:
        print(f"--- Hoja config: '{sheet_name}' ---")
        ws = find_worksheet_by_normalized_name(sh, sheet_name)
        if ws is None:
            print(f"  La hoja '{sheet_name}' NO existe (se crearía).")
            existing_last_tickers = []
        else:
            values = ws.get_all_values()
            if not values:
                existing_last_tickers = []
            else:
                header = values[0]
                try:
                    idx = [h.strip().lower() for h in header].index("ticker")
                except ValueError:
                    print("  ADVERTENCIA: no se encontró columna 'TICKER' en la hoja. Cabecera:", header)
                    existing_last_tickers = []
                else:
                    col = [row[idx] for row in values[1:] if len(row) > idx]
                    col_nonempty = [c for c in col if str(c).strip()!=""]
                    existing_last_tickers = col_nonempty[-last_n:]

        # Filtrado de candidates: comparar normalizado
        want_norm = normalize(sheet_name)
        # normalizamos la columna de sector y comparamos
        df_matches = df[[normalize(x) == want_norm for x in df[sector_col].astype(str).tolist()]]
        top_df = df_matches.head(top_n)
        top_tickers = [str(x).strip() for x in top_df['TICKER'].tolist()]
        to_add = [t for t in top_tickers if t not in existing_last_tickers]
        skipped = [t for t in top_tickers if t in existing_last_tickers]

        print(f"  top from CSV (up to {top_n}): {top_tickers}")
        print(f"  recientes en hoja (últimos {last_n}): {existing_last_tickers}")
        print(f"  -> se añadirían {len(to_add)}: {to_add}")
        if skipped:
            print(f"     (omitidos {len(skipped)} por duplicado reciente): {skipped}")
        print()

if __name__ == "__main__":
    main()
