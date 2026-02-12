#!/usr/bin/env python3
"""
run_pipeline.py

Orquestador simple y robusto para el pipeline:
  1) Ejecuta 1_fetch_indicators.py
  2) Ejecuta 2_score_select.py
  3) Ejecuta 3_export_sheets.py

Características:
 - Modos: full, fetch-only, score-only, export-only
 - Dry-run: no escribe (para comprobar pasos)
 - Logging a stdout y archivo run_pipeline.log
 - Reintentos controlables en cada etapa
 - Control de errores: por defecto aborta en error, opcional continue-on-error
 - Opcional: envía un resumen por email (placeholder) o imprime notificación
 - Uso: python run_pipeline.py --mode full
"""

import os
import sys
import subprocess
import argparse
import logging
import time
import shlex
from datetime import datetime

# CONFIG
LOG_FILE = "run_pipeline.log"
SCRIPTS = {
    "fetch": "1_fetch_indicators.py",
    "score": "2_score_select.py",
    "export": "3_export_sheets.py"
}

# Setup logging
logger = logging.getLogger("run_pipeline")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(fmt)
logger.addHandler(fh)


def call_script(script_path: str, timeout: int = 1800, dry_run: bool = False) -> int:
    """
    Ejecuta un script Python en un subproceso. Devuelve exit code.
    timeout en segundos; dry_run True => no ejecutar, solamente loggear.
    """
    if dry_run:
        logger.info("[DRY-RUN] Would run: %s", script_path)
        return 0
    if not os.path.exists(script_path):
        logger.error("Script no encontrado: %s", script_path)
        return 2
    cmd = [sys.executable, script_path]
    logger.info("Ejecutando: %s", " ".join(shlex.quote(p) for p in cmd))
    start = time.time()
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        dur = time.time() - start
        logger.info("Salida (%s): returncode=%d (t=%.1fs)", os.path.basename(script_path), res.returncode, dur)
        if res.stdout:
            logger.debug("stdout: %s", res.stdout.strip()[:4000])
        if res.stderr:
            logger.warning("stderr: %s", res.stderr.strip()[:4000])
        return res.returncode
    except subprocess.TimeoutExpired as e:
        logger.exception("Timeout: %s (after %s s)", script_path, timeout)
        return 124
    except Exception as e:
        logger.exception("Error ejecutando %s: %s", script_path, e)
        return 3


def run_stage(stage: str, retries: int, dry_run: bool, timeout: int, continue_on_error: bool) -> bool:
    """
    Ejecuta un stage (fetch/score/export) con reintentos.
    Devuelve True si succeed, False otherwise.
    """
    script = SCRIPTS.get(stage)
    if script is None:
        logger.error("Stage desconocido: %s", stage)
        return False

    attempt = 0
    while attempt <= retries:
        attempt += 1
        logger.info("Stage=%s attempt %d/%d", stage, attempt, retries + 1)
        code = call_script(script, timeout=timeout, dry_run=dry_run)
        if code == 0:
            logger.info("Stage %s completado correctamente.", stage)
            return True
        else:
            logger.warning("Stage %s fallo con code=%d", stage, code)
            if attempt <= retries - 0:
                backoff = min(60, 5 * attempt)
                logger.info("Reintentando en %ds...", backoff)
                time.sleep(backoff)
    # Si llegamos aquí falló todos los intentos
    logger.error("Stage=%s falló después de %d intentos.", stage, retries + 1)
    if continue_on_error:
        logger.warning("continue_on_error=True => continuando pipeline pese al fallo.")
        return False
    else:
        raise RuntimeError(f"Stage {stage} failed after {retries+1} attempts.")


def parse_args():
    p = argparse.ArgumentParser(description="Run pullback pipeline orchestrator")
    p.add_argument("--mode", choices=["full", "fetch-only", "score-only", "export-only"], default="full",
                   help="Modo de ejecución")
    p.add_argument("--dry-run", action="store_true", help="No ejecuta scripts; solo simula")
    p.add_argument("--retries", type=int, default=1, help="Reintentos por etapa (default 1)")
    p.add_argument("--timeout", type=int, default=1800, help="Timeout por script en segundos (default 1800)")
    p.add_argument("--continue-on-error", action="store_true", help="No abortar pipeline si una etapa falla")
    p.add_argument("--quiet", action="store_true", help="Menos log verbosity")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quiet:
        logger.setLevel(logging.WARNING)

    start_time = datetime.utcnow()
    logger.info("=== RUN PIPELINE START %s UTC ===", start_time.isoformat())

    try:
        if args.mode in ("full", "fetch-only"):
            run_stage("fetch", retries=args.retries, dry_run=args.dry_run, timeout=args.timeout, continue_on_error=args.continue_on_error)
        if args.mode in ("full", "score-only"):
            run_stage("score", retries=args.retries, dry_run=args.dry_run, timeout=args.timeout, continue_on_error=args.continue_on_error)
        if args.mode in ("full", "export-only"):
            run_stage("export", retries=args.retries, dry_run=args.dry_run, timeout=args.timeout, continue_on_error=args.continue_on_error)
    except Exception as e:
        logger.exception("Pipeline abortado: %s", e)
        logger.info("Revisa %s para más detalles.", LOG_FILE)
        # opcional: enviar notificación por email / webhook aquí
        sys.exit(1)

    end_time = datetime.utcnow()
    dur = (end_time - start_time).total_seconds()
    logger.info("=== RUN PIPELINE END %s UTC (duración %.1f s) ===", end_time.isoformat(), dur)
    # resumen: intenta leer intermediate/candidates.sheet_ready.csv y mostrar resumen
    try:
        import pandas as pd
        path = "intermediate/candidates.sheet_ready.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info("Resumen candidatos: total=%d (preview):", df.shape[0])
            logger.info(df.head(20).to_string(index=False))
    except Exception:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
