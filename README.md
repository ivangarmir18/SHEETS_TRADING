# Quant-Swing: Sistema de Soporte a la Decisión (DSS)

## Descripción General

**Quant-Swing** es un pipeline de ingeniería de datos diseñado para automatizar el análisis técnico y la gestión de riesgos en Swing Trading. El sistema actúa como un **Analista Cuantitativo Automatizado**, procesando cientos de activos (optimizados para QuantFury) para encontrar configuraciones de alta probabilidad.

El objetivo es eliminar la subjetividad emocional: el sistema calcula puntos de entrada, stops y objetivos basándose estrictamente en la volatilidad (ATR) y la tendencia matemática.

## Arquitectura del Proyecto

El sistema utiliza una arquitectura modular de "Separación de Responsabilidades":

* **Orquestador:** `run_pipeline.py` - Gestiona el flujo de ejecución.
* **Launcher:** `build_and_make_exe.bat` - Script de "Un Clic" que instala dependencias y ejecuta el programa.
* **Ingesta:** `1_fetch_indicators.py` - Descarga datos masivos y calcula indicadores (Macro EMAs, ATR, RSI).
* **Lógica:** `2_score_select.py` - Aplica el modelo de puntuación y filtra los mejores candidatos.
* **Exportación:** `3_export_sheets.py` - Conecta con la API de Google Sheets para subir los resultados.
* **Configuración:** `config.yaml` - Control central de parámetros de riesgo y listas de seguimiento.

## Requisitos Previos

1.  Python 3.8 o superior.
2.  Una cuenta de Google Cloud Platform (para la API de Sheets).
3.  Una Hoja de Cálculo de Google creada con las pestañas correspondientes (TECNOLOGÍA, SALUD, ENERGÍA, etc.).

## Instalación y Configuración

### 1. Configuración de Seguridad (Google Cloud)
El sistema necesita permiso para escribir en tu Excel.
1.  Obtén el archivo JSON de credenciales de una **Service Account** en Google Cloud.
2.  Crea una carpeta llamada `creds` en la raíz del proyecto.
3.  Renombra tu archivo a `gsheets-service.json` y mételo en esa carpeta.
4.  **Importante:** Comparte tu Google Sheet con el email de la Service Account (dándole permisos de Editor).

### 2. Configuración de Parámetros
Edita el archivo `config.yaml`:
* **spreadsheet_id:** Pega aquí el ID largo que aparece en la URL de tu Google Sheet.
* **sheets:** Asegúrate de que los nombres de la lista coinciden exactamente con las pestañas de tu Excel.

## Cómo Usar (Modo Automático)

No es necesario usar la terminal ni instalar librerías manualmente.

1.  Haz doble clic en el archivo **`build_and_make_exe.bat`**.
2.  El script verificará tu entorno, instalará las librerías necesarias (`pandas`, `yfinance`, `gspread`) y lanzará el análisis.
3.  Al finalizar, revisa tu Google Sheet. Los nuevos candidatos aparecerán al final de cada lista sin borrar tus operaciones anteriores.

## Lógica del "Edge" (Ventaja Estadística)

El sistema busca operaciones de **Reversión a la Media en Tendencia**:

1.  **Filtro de Tendencia:** Precio > EMA 150 (Temporalidad 2H).
2.  **Gestión de Riesgo (ATR):**
    * Entrada 1: Precio actual (o límite calculado).
    * Entrada 2: Entrada 1 - (1.0 x ATR).
    * Stop Loss: Calculado dinámicamente a 2x ATR para evitar ruido de mercado.
3.  **Scoring:** Se priorizan activos con divergencias en MFI (Money Flow Index) y RSI, indicando agotamiento de ventas en una tendencia alcista.

## Estructura de Carpetas

/raiz-del-proyecto
│
├── build_and_make_exe.bat  # EJECUTABLE PRINCIPAL
├── config.yaml             # Configuración de usuario
├── creds/                  # Carpeta de seguridad (Crear manualmente)
├── intermediate/           # Datos temporales (Parquet/CSV)
├── scripts .py             # Código fuente del sistema
└── README.md               # Este archivo

---
Desarrollado por Iván García Miranda
