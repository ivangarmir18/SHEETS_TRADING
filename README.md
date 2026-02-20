# SHEETS TRADING: Swing Trading ETL & DSS Pipeline

## 📌 Descripción General

**SHEETS TRADING** es un pipeline ETL (Extract, Transform, Load) desarrollado en Python. Actúa como el motor de un Sistema de Soporte a la Decisión (DSS) diseñado para automatizar el análisis cuantitativo y la gestión de riesgos en operativas de swing trading rápido (1-3 días).

El objetivo del proyecto es eliminar el trabajo manual de recolección de datos y cálculo de métricas (como ATR, EMAs, RSI y proyecciones de riesgo), alimentando de forma totalmente automatizada un panel de control en Google Sheets. Esto permite calcular al instante bloques de entradas escalonadas, stops dinámicos y targets precisos para plataformas de ejecución.

> **🤖 Nota sobre el desarrollo asistido por IA:** > La arquitectura del pipeline, la lógica de negocio, la separación de responsabilidades (ETL) y el diseño de la automatización son de mi autoría. Para la redacción de la sintaxis pura y la optimización algorítmica del código en Python, me he apoyado intensivamente en Inteligencia Artificial. Mi enfoque en este proyecto es de arquitecto, definiendo el problema, estructurando los módulos y auditando la lógica, delegando el "picado de código" a la IA para maximizar la eficiencia.

## ⚙️ Arquitectura del Sistema (ETL Pipeline)

El sistema está diseñado bajo una arquitectura modular, optimizada para el procesamiento concurrente y la eficiencia de red.

### 1. Extract (`1_fetch_indicators.py`)
* **Ingesta masiva:** Utiliza la API de `yfinance` para descargar datos intradiarios de cientos de activos financieros clasificados por sectores.
* **Optimización de rendimiento:** Implementa `ProcessPoolExecutor` para paralelizar las peticiones HTTP y los cálculos matemáticos.
* **Sistema de Caché:** Integra un mecanismo de caché local (TTL configurable) para evitar peticiones redundantes y bloqueos de la API externa.

### 2. Transform (`2_score_select.py`)
* **Procesamiento vectorial:** Uso intensivo de `pandas` y `numpy` para el cálculo de indicadores técnicos complejos (Macro EMAs, Volatilidad ATR, Divergencias RSI/MFI).
* **Filtros de Negocio:** Aplica un sistema de *scoring* dinámico basado en algoritmos de reversión a la media. Filtra los activos evaluando distancias respecto a medias móviles y umbrales de agotamiento de volumen.

### 3. Load (`3_export_sheets.py`)
* **Integración Cloud:** Conexión autenticada mediante Google Cloud Platform (Service Accounts) a la API de Google Sheets (`gspread`).
* **Actualización en Bloque:** Los datos filtrados y puntuados se estructuran y se envían mediante operaciones *batch* (en bloque) a las pestañas correspondientes del Excel, minimizando las cuotas de uso de la API y garantizando la persistencia de datos históricos.

### Orquestación y Control
* **`run_pipeline.py` & `gui_launcher.py`:** Scripts orquestadores (CLI y GUI) que gestionan el flujo de ejecución, el manejo de excepciones, los reintentos automáticos y el registro de eventos (*logging*).
* **`config.yaml`:** Archivo centralizado para la configuración de parámetros de riesgo, ponderación de algoritmos y listas de activos.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.8+
* **Procesamiento de Datos:** Pandas, Numpy
* **Concurrencia:** `concurrent.futures` (Multiprocessing)
* **Integración API:** Google Cloud API, `gspread`, `yfinance`
* **Formatos de datos:** Parquet (almacenamiento intermedio de alta velocidad), CSV, YAML.

## 📊 Lógica de Negocio (El "Edge" Estadístico)

El modelo matemático detrás del pipeline busca automatizar la detección de configuraciones de alta probabilidad basándose estrictamente en la volatilidad:
* **Filtro de Tendencia Macro:** Precio > EMA 150 en temporalidades de 12H.
* **Gestión de Riesgo por Volatilidad (ATR):** Cálculo automatizado de la distancia del ATR para definir rangos de entrada precisos y Stops Loss dinámicos, aislando el ruido del mercado.
* **Scoring por Divergencias:** Asignación de pesos matemáticos a activos que muestran discrepancias entre el flujo de capital (MFI) y la fuerza relativa (RSI).

## 🚀 Instalación y Despliegue Local

Para auditar o ejecutar este código localmente, se requiere configuración de credenciales Cloud:

1. Clonar el repositorio e instalar dependencias necesarias ya integradas en el propio build_and_make_exe.bat (Asegúrate de tener instalados pandas, numpy, yfinance, gspread, pyyaml).

2. Crear un proyecto en Google Cloud Console, habilitar la Google Sheets API y generar una clave de cuenta de servicio (Service Account).

3. Guardar el archivo JSON generado en creds/gsheets-service.json.

4. Ajustar los parámetros de ponderación y el spreadsheet_id en config.yaml.

5. Ejecutar el orquestador:
python run_pipeline.py --mode full
(O alternativamente, usar el lanzador visual python gui_launcher.py)

Desarrollado por Iván García Miranda.
