#!/usr/bin/env python3
# gui_launcher.py
"""
Robust Launcher for the pullback pipeline.

Features:
 - Two tabs: Control (run pipeline, logs, utilities) and Settings (edit config.yaml inline).
 - Buttons: Fetch, Score, Export, Full, Run Selected Stage, Stop, Clean Cache, Open folder, Export logs.
 - Settings panel: sectors (checkbox list, from sectors.csv if present), top_n_per_sector,
   presets (conservative/balanced/aggressive), manual edit of soft-weights, Save/Backup/Restore.
 - Save & Run from settings.
 - Live stdout streaming into a log panel and log file saved at logs/run_YYYYmmdd_HHMMSS.log.
 - Stop/killer for running subprocess.
 - Summary reader from intermediate/candidates.sheet_ready.csv to preview candidates.
 - Cross-platform compatibility.
"""
from __future__ import annotations

import os
import sys
import yaml
import threading
import subprocess
import shlex
import time
import shutil
from datetime import datetime
from typing import List, Dict, Optional

# Tkinter UI
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

# Optional pandas usage for CSV reading (sectors, candidates)
try:
    import pandas as pd
except Exception:
    pd = None

# Defaults (same keys used by other scripts)
DEFAULT_SHEETS = ["energía", "salud", "commodites", "financiero", "tecnología"]
DEFAULT_WEIGHTS = {
    'rsi_low': 20,
    'mfi_rsi_soft': 15,
    'macd_pos': 10,
    'ema15_close_gap': 20,
    'atr_score': 20,
    'ema150_distance': 10,
    'ema50_below': 5
}
PRESETS = {
    'conservative': {
        'rsi_low': 25,
        'mfi_rsi_soft': 20,
        'macd_pos': 5,
        'ema15_close_gap': 20,
        'atr_score': 10,
        'ema150_distance': 15,
        'ema50_below': 5
    },
    'balanced': DEFAULT_WEIGHTS.copy(),
    'aggressive': {
        'rsi_low': 15,
        'mfi_rsi_soft': 20,
        'macd_pos': 15,
        'ema15_close_gap': 25,
        'atr_score': 10,
        'ema150_distance': 10,
        'ema50_below': 5
    }
}

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
SECTORS_CSV = os.path.join(ROOT, "sectors.csv")
RUN_PIPELINE = os.path.join(ROOT, "run_pipeline.py")
CANDIDATES_SHEET_READY = os.path.join(ROOT, "intermediate", "candidates.sheet_ready.csv")
INDICATORS_PARQUET = os.path.join(ROOT, "intermediate", "indicators.parquet")
CACHE_DIR = os.path.join(ROOT, "intermediate", "cache")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Stage commands (run_pipeline orchestrates; still allow direct script calls)
CMD_MAP = {
    "fetch": [sys.executable, "run_pipeline.py", "--mode", "fetch-only"],
    "score": [sys.executable, "run_pipeline.py", "--mode", "score-only"],
    "export": [sys.executable, "run_pipeline.py", "--mode", "export-only"],
    "full": [sys.executable, "run_pipeline.py", "--mode", "full"],
}

# --------------- Helpers ---------------
def load_config(path: str = CONFIG_PATH) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}

def save_config(cfg: Dict, path: str = CONFIG_PATH) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

def backup_config(path: str = CONFIG_PATH) -> Optional[str]:
    if not os.path.exists(path):
        return None
    bak = f"{path}.bak"
    try:
        shutil.copy2(path, bak)
        return bak
    except Exception:
        return None

def restore_config_from_backup(path: str = CONFIG_PATH) -> bool:
    bak = f"{path}.bak"
    if not os.path.exists(bak):
        return False
    try:
        shutil.copy2(bak, path)
        return True
    except Exception:
        return False

def read_sectors_csv(path: str = SECTORS_CSV) -> List[str]:
    if pd is None or not os.path.exists(path):
        return _unique_preserve_order(DEFAULT_SHEETS.copy())
    try:
        df = pd.read_csv(path, dtype=str)
        if 'Sector' in df.columns:
            vals = [str(x).strip() for x in df['Sector'].dropna().tolist()]
            return _unique_preserve_order(vals) if vals else _unique_preserve_order(DEFAULT_SHEETS.copy())
    except Exception:
        pass
    return _unique_preserve_order(DEFAULT_SHEETS.copy())

def _unique_preserve_order(lst: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in lst:
        key = v.strip()
        if not key:
            continue
        if key.lower() not in seen:
            seen.add(key.lower())
            out.append(key)
    return out

def read_candidates_preview(path: str = CANDIDATES_SHEET_READY) -> Optional[pd.DataFrame]:
    if pd is None or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return None

def tidy_cmd(cmd_list: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in cmd_list)

def write_log_file(prefix="run"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fn = f"{prefix}_{ts}.log"
    return os.path.join(LOG_DIR, fn)

# --------------- UI ---------------
class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pullback Pipeline - Launcher (robust)")
        self.geometry("1100x720")
        self.minsize(900, 620)

        # Process handle and log file
        self.proc: Optional[subprocess.Popen] = None
        self.logfile_path: Optional[str] = None
        self._proc_thread: Optional[threading.Thread] = None

        # Load initial config if any
        self.cfg = load_config()
        self.sectors_list = read_sectors_csv()

        # Default UI values
        self.selected_stage = tk.StringVar(value="full")
        self.dry_run = tk.IntVar(value=0)
        self.retries = tk.IntVar(value=int(self.cfg.get("retries", 1)))
        self.timeout = tk.IntVar(value=int(self.cfg.get("timeout", 1800)))
        self.continue_on_error = tk.IntVar(value=1 if self.cfg.get("continue_on_error") else 0)

        # Settings UI state
        self.sheets_selected_vars: Dict[str, tk.IntVar] = {}
        self.top_n_var = tk.IntVar(value=int(self.cfg.get("top_n_per_sector", 4)))
        # weights
        weights_cfg = self.cfg.get("weights", DEFAULT_WEIGHTS.copy())
        for k in DEFAULT_WEIGHTS.keys():
            weights_cfg.setdefault(k, DEFAULT_WEIGHTS[k])
        self.weights_vars: Dict[str, tk.StringVar] = {k: tk.StringVar(value=str(weights_cfg.get(k))) for k in DEFAULT_WEIGHTS.keys()}
        self.preset_var = tk.StringVar(value=self._detect_preset_name(weights_cfg))

        # Build UI (tabs)
        self._build_ui()

    # ---------- build UI ----------
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # Controls tab
        tab_controls = ttk.Frame(nb)
        nb.add(tab_controls, text="Control")

        # Settings tab
        tab_settings = ttk.Frame(nb)
        nb.add(tab_settings, text="Settings / Config")

        self._build_controls_tab(tab_controls)
        self._build_settings_tab(tab_settings)

        # bottom log area
        bottom = tk.Frame(self, bd=1, relief="sunken")
        bottom.pack(fill="both", expand=False, padx=8, pady=(0,8))
        lbl = tk.Label(bottom, text="Output / Live Logs:", font=("Arial", 10, "bold"))
        lbl.pack(anchor="w")
        self.log_txt = ScrolledText(bottom, height=10, font=("Consolas", 10))
        self.log_txt.pack(fill="both", expand=True)
        self._log("Launcher ready.")

    # ---------- Controls tab ----------
    def _build_controls_tab(self, parent):
        frm_top = tk.Frame(parent)
        frm_top.pack(fill="x", pady=6, padx=6)

        btn_font = ("Arial", 11, "bold")
        # stage quick buttons
        btns = [
            ("Fetch", "fetch"),
            ("Score", "score"),
            ("Export", "export"),
            ("Full", "full"),
        ]
        for (label, stage) in btns:
            b = tk.Button(frm_top, text=label, font=btn_font, width=14, height=2,
                          command=lambda s=stage: self._start_run_stage_cmd(s))
            b.pack(side="left", padx=6)

        # right side controls
        right = tk.Frame(frm_top)
        right.pack(side="right", fill="y")

        # stage selection and run
        sel_frame = tk.Frame(parent, bd=1, relief="groove", padx=8, pady=8)
        sel_frame.pack(fill="x", padx=6, pady=6)
        tk.Label(sel_frame, text="Run selected stage:", font=("Arial", 10, "bold")).pack(anchor="w")
        stage_select = ttk.Combobox(sel_frame, values=list(CMD_MAP.keys()), textvariable=self.selected_stage, state="readonly")
        stage_select.pack(side="left", padx=(0,8))
        tk.Checkbutton(sel_frame, text="Dry-run", variable=self.dry_run).pack(side="left", padx=6)
        tk.Label(sel_frame, text="Retries:").pack(side="left", padx=(12,2))
        tk.Spinbox(sel_frame, from_=0, to=10, width=4, textvariable=self.retries).pack(side="left")
        tk.Label(sel_frame, text="Timeout(s):").pack(side="left", padx=(12,2))
        tk.Spinbox(sel_frame, from_=60, to=86400, width=8, textvariable=self.timeout).pack(side="left")
        tk.Checkbutton(sel_frame, text="Continue on error", variable=self.continue_on_error).pack(side="left", padx=12)

        run_row = tk.Frame(parent)
        run_row.pack(fill="x", padx=6, pady=(4,6))
        tk.Button(run_row, text="Run Selected Stage", font=btn_font, bg="#1976D2", fg="white",
                  command=self.on_run_selected_stage).pack(side="left", padx=6)
        tk.Button(run_row, text="Stop running process", font=btn_font, bg="#D32F2F", fg="white",
                  command=self.on_stop_process).pack(side="left", padx=6)
        tk.Button(run_row, text="Open intermediate folder", command=self.open_intermediate_folder).pack(side="right", padx=6)
        tk.Button(run_row, text="Export Logs...", command=self.export_logs_dialog).pack(side="right", padx=6)

        # quick preview candidates
        preview_frame = tk.Frame(parent, bd=1, relief="groove", padx=8, pady=8)
        preview_frame.pack(fill="x", padx=6, pady=6)
        tk.Label(preview_frame, text="Candidatos (sheet_ready) preview:", font=("Arial", 10, "bold")).pack(anchor="w")
        btn_preview = tk.Button(preview_frame, text="Show candidates summary", command=self.show_candidates_summary)
        btn_preview.pack(anchor="w", pady=6)

        # clean cache
        util_frame = tk.Frame(parent)
        util_frame.pack(fill="x", padx=6, pady=(6,10))
        tk.Button(util_frame, text="Clean Cache (intermediate/cache)", command=self.on_clean_cache).pack(side="left", padx=6)
        tk.Button(util_frame, text="Backup config.yaml", command=self.on_backup_config).pack(side="left", padx=6)
        tk.Button(util_frame, text="Restore config.yaml.bak", command=self.on_restore_config).pack(side="left", padx=6)

    # ---------- Settings tab ----------
    def _build_settings_tab(self, parent):
        left = tk.Frame(parent, bd=1, relief="groove")
        left.place(x=8, y=8, width=520, height=520)

        right = tk.Frame(parent, bd=1, relief="groove")
        right.place(x=540, y=8, width=520, height=520)

        # Left: sectors + top_n
        tk.Label(left, text="Sectores (selecciona):", font=("Arial", 12, "bold")).pack(anchor="w", padx=8, pady=(8,4))
        # scroll area
        canvas = tk.Canvas(left, borderwidth=0, highlightthickness=0)
        box = tk.Frame(canvas)
        vscroll = tk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        canvas.create_window((0, 0), window=box, anchor="nw")

        def onconfig(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        box.bind("<Configure>", onconfig)

        # load current sheets from config or default
        cfg_sheets = self.cfg.get("sheets", self.sectors_list)
        # if config has values, dedupe and use in order
        cfg_sheets = _unique_preserve_order(list(cfg_sheets))

        for i, s in enumerate(self.sectors_list):
            var = tk.IntVar(value=1 if s in cfg_sheets else 0)
            cb = tk.Checkbutton(box, text=s, variable=var, font=("Arial", 10))
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=6, pady=4)
            self.sheets_selected_vars[s] = var

        btns = tk.Frame(left)
        btns.pack(fill="x", padx=8, pady=6)
        tk.Button(btns, text="Select all", command=self._select_all_sheets).pack(side="left", padx=6)
        tk.Button(btns, text="Deselect all", command=self._deselect_all_sheets).pack(side="left", padx=6)

        tk.Label(left, text="Numero de trades por sector (top_n_per_sector):", font=("Arial", 11, "bold")).pack(anchor="w", padx=8, pady=(8,4))
        spin = tk.Spinbox(left, from_=1, to=50, textvariable=self.top_n_var, width=6)
        spin.pack(anchor="w", padx=8)

        # Save / Save & Run
        tk.Button(left, text="Save config.yaml", bg="#4CAF50", fg="white", command=self.on_save_config).pack(side="left", padx=8, pady=10)
        tk.Button(left, text="Save & Run pipeline", bg="#1976D2", fg="white", command=self.on_save_and_run).pack(side="left", padx=8, pady=10)

        # Right: presets + weights
        tk.Label(right, text="Ponderaciones soft (presets):", font=("Arial", 12, "bold")).pack(anchor="w", padx=8, pady=(8,4))
        presets = list(PRESETS.keys()) + ["custom"]
        cmb = ttk.Combobox(right, values=presets, textvariable=self.preset_var, state="readonly")
        cmb.pack(anchor="w", padx=8)

        tk.Label(right, text="Ponderaciones actuales (edítalas si quieres):", font=("Arial", 10)).pack(anchor="w", padx=8, pady=(8,4))
        weights_frame = tk.Frame(right)
        weights_frame.pack(fill="both", padx=8)
        self.weights_entries: Dict[str, tk.Entry] = {}
        for i, key in enumerate(DEFAULT_WEIGHTS.keys()):
            lbl = tk.Label(weights_frame, text=key, width=20, anchor="w")
            lbl.grid(row=i, column=0, padx=4, pady=4)
            ent = tk.Entry(weights_frame, width=8, textvariable=self.weights_vars[key])
            ent.grid(row=i, column=1, padx=4, pady=4)
            self.weights_entries[key] = ent

        wp = tk.Frame(right)
        wp.pack(fill="x", pady=6, padx=8)
        tk.Button(wp, text="Apply preset", command=self.on_apply_preset).pack(side="left", padx=6)
        tk.Button(wp, text="Reset to balanced", command=self.on_reset_balanced).pack(side="left", padx=6)
        tk.Button(wp, text="Preview config", command=self.on_preview_config).pack(side="right", padx=6)

        # backup and restore
        br = tk.Frame(right)
        br.pack(fill="x", padx=8, pady=(10,4))
        tk.Button(br, text="Backup config.yaml (.bak)", command=self.on_backup_config).pack(side="left", padx=6)
        tk.Button(br, text="Restore config.yaml.bak", command=self.on_restore_config).pack(side="left", padx=6)

    # ---------- helpers: Settings ----------
    def _select_all_sheets(self):
        for v in self.sheets_selected_vars.values():
            v.set(1)

    def _deselect_all_sheets(self):
        for v in self.sheets_selected_vars.values():
            v.set(0)

    def _collect_config_from_ui(self) -> Dict:
        sheets = [s for s, v in self.sheets_selected_vars.items() if v.get() == 1]
        top_n = int(self.top_n_var.get())
        weights = {}
        for k, var in self.weights_vars.items():
            try:
                weights[k] = int(var.get())
            except Exception:
                try:
                    weights[k] = int(float(var.get()))
                except Exception:
                    weights[k] = DEFAULT_WEIGHTS.get(k, 0)
        cfg_out = load_config()
        cfg_out['sheets'] = sheets
        cfg_out['top_n_per_sector'] = top_n
        cfg_out['weights'] = weights
        # keep existing keys like indicators_parquet / candidates_parquet if present
        return cfg_out

    def on_apply_preset(self):
        name = self.preset_var.get()
        if name in PRESETS:
            preset = PRESETS[name]
            for k, v in preset.items():
                if k in self.weights_vars:
                    self.weights_vars[k].set(str(v))
            self._log(f"Applied preset '{name}'.")
        else:
            self._log("Preset 'custom' selected; edit weights manually.")

    def on_reset_balanced(self):
        preset = PRESETS['balanced']
        for k, v in preset.items():
            if k in self.weights_vars:
                self.weights_vars[k].set(str(v))
        self.preset_var.set('balanced')
        self._log("Reset to balanced preset.")

    def on_save_config(self):
        cfg_out = self._collect_config_from_ui()
        errs = []
        if not cfg_out.get('sheets'):
            errs.append("Seleccione al menos 1 sector.")
        if int(cfg_out.get('top_n_per_sector', 0)) <= 0:
            errs.append("top_n_per_sector debe ser entero > 0.")
        for k, v in cfg_out.get('weights', {}).items():
            if not isinstance(v, int) or v < 0:
                errs.append(f"Peso inválido para {k}: {v}")

        if errs:
            messagebox.showerror("Errores en configuración", "\n".join(errs))
            return
        # backup existing
        bak = backup_config(CONFIG_PATH)
        if bak:
            self._log(f"Backup of existing config created: {bak}")
        try:
            save_config(cfg_out, CONFIG_PATH)
            self._log(f"Saved config.yaml ({CONFIG_PATH})")
            messagebox.showinfo("Guardado", f"config.yaml guardado en {CONFIG_PATH}")
        except Exception as e:
            messagebox.showerror("Error guardando", str(e))
            self._log(f"Error saving config: {e}")

    def on_save_and_run(self):
        self.on_save_config()
        # then run full
        self._start_run_stage_cmd("full")

    def on_backup_config(self):
        bak = backup_config(CONFIG_PATH)
        if bak:
            messagebox.showinfo("Backup", f"Backup creado: {bak}")
            self._log(f"Backup created: {bak}")
        else:
            messagebox.showwarning("Backup", "No se pudo crear backup (maybe config.yaml missing)")

    def on_restore_config(self):
        ok = restore_config_from_backup(CONFIG_PATH)
        if ok:
            messagebox.showinfo("Restore", "config.yaml restaurado desde config.yaml.bak")
            self._log("config.yaml restored from backup. You may want to restart the app to reload settings.")
        else:
            messagebox.showwarning("Restore", "No se encontró config.yaml.bak")

    def on_preview_config(self):
        cfg = self._collect_config_from_ui()
        pretty = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        PreviewDialog(self, pretty)

    # ---------- run pipeline operations ----------
    def _start_run_stage_cmd(self, stage: str):
        # Builds base cmd and runs with configured options
        if stage not in CMD_MAP:
            messagebox.showerror("Stage error", f"Unknown stage: {stage}")
            return
        # Save config first if we have any unsaved changes? We don't force it, but good to save current UI state to config
        try:
            self.on_save_config()
        except Exception:
            pass

        # Build run_pipeline arguments to respect retries/timeout/continue_on_error/dry-run
        # run_pipeline supports flags in its parse_args (we used earlier). We'll pass them as env or args.
        cmd = CMD_MAP[stage].copy()
        # Add extras as flags
        if self.dry_run.get():
            cmd.append("--dry-run")
        cmd.extend(["--retries", str(self.retries.get())])
        cmd.extend(["--timeout", str(self.timeout.get())])
        if self.continue_on_error.get():
            cmd.append("--continue-on-error")
        # start background thread to run and stream output
        self._start_process_thread(cmd)

    def on_run_selected_stage(self):
        self._start_run_stage_cmd(self.selected_stage.get())

    def _start_process_thread(self, cmd_list: List[str]):
        # ensure not already running
        if self.proc and (self.proc.poll() is None):
            messagebox.showwarning("Running", "A process is already running. Stop it first.")
            return
        # prepare logfile
        self.logfile_path = write_log_file("pipeline")
        self._log(f"Log file: {self.logfile_path}")
        t = threading.Thread(target=self._run_and_stream, args=(cmd_list, self.logfile_path), daemon=True)
        t.start()
        self._proc_thread = t

    def _run_and_stream(self, cmd_list: List[str], logfile: str):
        try:
            self._disable_all_controls(True)
            self._log(f"Starting: {tidy_cmd(cmd_list)}")
            with open(logfile, "w", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd_list, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                self.proc = proc
                # stream
                for line in iter(proc.stdout.readline, ""):
                    if line is None:
                        break
                    out = line.rstrip("\n")
                    fh.write(out + "\n")
                    fh.flush()
                    self._log(out)
                proc.wait()
                ret = proc.returncode
                fh.write(f"\nPROCESS EXIT CODE: {ret}\n")
                fh.flush()
                self._log(f"Process finished with code {ret}")
                # show info when done
                messagebox.showinfo("Process finished", f"Process finished with code {ret}\nLog: {logfile}")
        except Exception as e:
            self._log(f"Exception running process: {e}")
            messagebox.showerror("Run error", str(e))
        finally:
            self.proc = None
            self._disable_all_controls(False)

    def on_stop_process(self):
        if not self.proc:
            messagebox.showinfo("No process", "No process is currently running.")
            return
        if self.proc.poll() is None:
            try:
                self._log("Attempting to terminate process...")
                self.proc.terminate()
                # wait a bit
                t0 = time.time()
                while self.proc.poll() is None and (time.time() - t0) < 5:
                    time.sleep(0.2)
                if self.proc.poll() is None:
                    self._log("Killing process...")
                    self.proc.kill()
                self._log("Process terminated.")
            except Exception as e:
                self._log(f"Error stopping process: {e}")
                messagebox.showwarning("Stop error", str(e))
        else:
            messagebox.showinfo("Not running", "Process already finished.")

    # ---------- utilities ----------
    def on_clean_cache(self):
        if not os.path.exists(CACHE_DIR):
            messagebox.showinfo("Cache", "No cache directory found.")
            return
        if not messagebox.askyesno("Confirm", f"Delete cache folder: {CACHE_DIR}?"):
            return
        t = threading.Thread(target=self._clean_cache_worker, daemon=True)
        t.start()

    def _clean_cache_worker(self):
        self._disable_all_controls(True)
        deleted = 0
        try:
            for root, dirs, files in os.walk(CACHE_DIR):
                for f in files:
                    p = os.path.join(root, f)
                    try:
                        os.remove(p)
                        deleted += 1
                    except Exception:
                        pass
            # try to remove empty dirs beneath cache dir
            for root, dirs, files in os.walk(CACHE_DIR, topdown=False):
                for d in dirs:
                    p = os.path.join(root, d)
                    try:
                        os.rmdir(p)
                    except Exception:
                        pass
            self._log(f"Cache cleaned. Files deleted: {deleted}")
            messagebox.showinfo("Cache cleaned", f"Files deleted: {deleted}")
        finally:
            self._disable_all_controls(False)

    def open_intermediate_folder(self):
        path = os.path.join(ROOT, "intermediate")
        if not os.path.exists(path):
            messagebox.showwarning("Folder not found", f"{path} not found.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Open folder error", str(e))

    def export_logs_dialog(self):
        # Save current visible logs to a file accessible for user (just copy logfile or create a new file)
        if not self.logfile_path or not os.path.exists(self.logfile_path):
            # create temporary log from current content
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            outp = os.path.join(LOG_DIR, f"interactive_log_{ts}.log")
            try:
                with open(outp, "w", encoding="utf-8") as fh:
                    fh.write(self.log_txt.get("1.0", "end"))
                messagebox.showinfo("Logs exported", f"Logs saved to {outp}")
            except Exception as e:
                messagebox.showerror("Export error", str(e))
        else:
            messagebox.showinfo("Log file", f"Log file path: {self.logfile_path}")

    def show_candidates_summary(self):
        df = read_candidates_preview()
        if df is None:
            messagebox.showinfo("Candidates", "No candidates.sheet_ready.csv found or pandas not available.")
            return
        # show top 20 rows in a small dialog
        preview = df.head(20).to_string(index=False)
        PreviewDialog(self, preview, title="Candidates preview (top 20)")

    # ---------- utilities: UI state ----------
    def _disable_all_controls(self, disable: bool):
        # disable/enable main interactive widgets
        # Iterate over children and disable buttons / inputs conservatively
        for w in self.winfo_children():
            try:
                for c in w.winfo_children():
                    try:
                        c.config(state=("disabled" if disable else "normal"))
                    except Exception:
                        pass
            except Exception:
                pass
        # but keep log area enabled always
        try:
            self.log_txt.config(state="normal")
        except Exception:
            pass

    def _log(self, text: str):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.log_txt.insert("end", f"[{ts}] {text}\n")
        self.log_txt.see("end")

    def _detect_preset_name(self, weights: Dict[str, int]) -> str:
        for name, preset in PRESETS.items():
            if all(int(preset.get(k, 0)) == int(weights.get(k, 0)) for k in preset.keys()):
                return name
        return "custom"

# ---------- small dialogs ----------
class PreviewDialog(tk.Toplevel):
    def __init__(self, master, text: str, title="Preview"):
        super().__init__(master)
        self.title(title)
        self.geometry("800x480")
        txt = ScrolledText(self, font=("Courier New", 10))
        txt.insert("1.0", text)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)
        btn = tk.Button(self, text="Close", command=self.destroy)
        btn.pack(pady=6)

# --------------- run ---------------
def main():
    app = LauncherApp()
    app.mainloop()

if __name__ == "__main__":
    main()
