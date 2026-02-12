```markdown
# 📈 Sistema de Soporte a la Decisión (DSS) para Trading

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas) ![Google Cloud](https://img.shields.io/badge/Google_Cloud-Sheets_API-4285F4?style=for-the-badge&logo=google-cloud) ![Platform](https://img.shields.io/badge/Platform-QuantFury-red?style=for-the-badge)

## 📋 Descripción General

Este proyecto es un pipeline de ingeniería de datos modular diseñado para automatizar el análisis técnico y la gestión de riesgos en estrategias de Swing Trading. Optimizado específicamente para el universo de activos de **QuantFury**, este sistema actúa como un **DSS (Decision Support System)**, filtrando cientos de activos para identificar configuraciones de alta probabilidad basadas en anomalías estadísticas y alineación de tendencias.

A diferencia de los *screeners* convencionales, este sistema implementa un **modelo de puntuación ponderada** que combina volatilidad (ATR), momentum (RSI/MFI) y jerarquía de tendencias (Macro EMAs), exportando las operaciones procesables directamente a un panel de control en la nube (Google Sheets).

## 🏗 Arquitectura del Sistema

El proyecto sigue un patrón de diseño de **Separación de Responsabilidades (SoC)** para asegurar la escalabilidad y facilitar la depuración:

```text
├── 📂 root
│   ├── 📜 run_pipeline.py        # Orquestador: Gestiona el flujo de ejecución y manejo de errores
│   ├── 📜 1_fetch_indicators.py  # Capa de Ingesta: Descarga por lotes (yfinance) y cálculo de indicadores
│   ├── 📜 2_score_select.py      # Capa Lógica: Modelo de scoring multifactorial y filtrado de candidatos
│   ├── 📜 3_export_sheets.py     # Capa de Presentación: Integración con Google Sheets API
│   ├── 📜 config.yaml            # Configuración: Parámetros centralizados (Riesgo, APIs, Tickers)
│   ├── ⚙️ build_and_make_exe.bat # Launcher: Instalación de dependencias y ejecución automática
│   ├── 📂 creds/                 # Seguridad: Llaves de Servicio de Google Cloud (GitIgnored)
│   └── 📂 intermediate/          # Almacenamiento: Archivos Parquet/CSV para persistencia entre procesos

```

## 🚀 Características Clave

### 1. Ingesta de Datos de Alto Rendimiento

* Utiliza `ProcessPoolExecutor` para la **carga concurrente** de cientos de tickers.
* Implementa **Análisis de Macro EMAs** (EMA 400/500/600 en temporalidad de 2H) para contextualizar tendencias seculares.

### 2. Gestión de Riesgo Algorítmica

El sistema no solo señala "compra/venta"; calcula la estructura exacta de la operación basada en la volatilidad:

* **Stops Dinámicos:** El Stop Loss se calcula como un múltiplo del **ATR (Average True Range)**.
* **Optimización DCA:** Calcula puntos de entrada escalonados (`Entrada 1` y `Entrada 2`) para optimizar el precio promedio durante los retrocesos.

### 3. Modelo de Scoring Multifactorial

Los candidatos son clasificados, no solo filtrados. El algoritmo premia:

* **Alineación de Tendencia:** Precio > EMA 150.
* **Potencial de Reversión:** Gap (distancia) entre Precio y EMA 15.
* **Divergencia de Volumen/Momentum:** Lógica avanzada que compara MFI (Money Flow Index) vs RSI para detectar agotamiento.

### 4. Integración en la Nube

* **Exportación No Destructiva:** Añade nuevos candidatos a pestañas específicas de Google Sheets (ej. TECNOLOGÍA, ENERGÍA) sin sobrescribir el historial.
* **Formato Inteligente:** Aplica formato condicional y estilos automáticamente vía API.

---

## 🛠️ Instalación y Configuración

### Prerrequisitos

* Python 3.8 o superior instalado en el sistema.
* Un proyecto en Google Cloud Platform (GCP) con la **Google Sheets API** habilitada.

### 1. Clonar el Repositorio

```bash
git clone [https://github.com/tuusuario/quant-swing-dss.git](https://github.com/tuusuario/quant-swing-dss.git)
cd quant-swing-dss

```

### 2. Configuración de Credenciales (CRÍTICO)

Para permitir la exportación a la nube, necesitas una llave de Cuenta de Servicio (Service Account).

1. Crea una Service Account en GCP.
2. Descarga el archivo de claves JSON.
3. **Crea una carpeta llamada `creds**` en la raíz del proyecto.
4. Renombra tu llave a `gsheets-service.json` y colócala dentro de `creds/`.

> **IMPORTANTE:** La carpeta `creds/` está añadida al `.gitignore` para evitar filtrar claves privadas.

### 3. Configuración (`config.yaml`)

Configura el ID de tu hoja de cálculo y los parámetros de riesgo en `config.yaml`:

```yaml
spreadsheet_id: "TU_ID_DE_GOOGLE_SHEETS_AQUI"
sheets:
  - TECHNOLOGY
  - ENERGY
  - HEALTH
  - FINANCIAL
risk_parameters:
  min_atr_distance: 0.1
  rsi_2h_max: 60

```

---

## ⚡ Ejecución (Launcher Automático)

El proyecto incluye un script `.bat` que gestiona automáticamente el entorno. No es necesario instalar librerías manualmente.

1. Localiza el archivo **`build_and_make_exe.bat`**.
2. Haz doble clic para ejecutarlo.

**¿Qué hace este script?**

* Verifica e instala/actualiza automáticamente todas las dependencias necesarias (`pandas`, `yfinance`, `gspread`, etc.).
* Lanza la interfaz gráfica (`gui_launcher.py`) o el orquestador del pipeline.

---

## 📊 Lógica y Fórmulas (El "Edge")

La ventaja estadística del sistema se basa en entradas ajustadas por volatilidad:

* **Zona de Entrada:** 
* **Filtro de Tendencia Duro:**  (Solo Largos)
* **Gatillo de Momentum:**  Y  (comprando debilidad en tendencia alcista)

---

## ⚠️ Disclaimer

Este software es para **fines educativos y de investigación únicamente**. Actúa como una herramienta para filtrar el ruido del mercado y no constituye asesoramiento financiero. Operar en mercados financieros (Cripto, Acciones, Forex) conlleva un alto nivel de riesgo.

---

*Desarrollado por Iván García Miranda*

