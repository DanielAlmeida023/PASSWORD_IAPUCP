# INTI-3K · Alerta de heladas a 3 km para la Cordillera Blanca

**Hackathon IA PUCP 2026 (DSC PUCP × IA-PUCP) · Equipo `PASSWORD`**

<!-- TODO EQUIPO: pegar el link PÚBLICO del video (YouTube no listado / Drive "cualquiera con el link") -->
- **Video pitch (≤ 5 min):** `>>> TODO EQUIPO: LINK DEL VIDEO <<<`
<!-- TODO EQUIPO: nombres completos de los integrantes (2 a 4) -->
- **Integrantes:** `>>> TODO EQUIPO: Nombre 1, Nombre 2, Nombre 3, Nombre 4 <<<`
- **Reporte para el jurado:** [`REPORTE_PRODUCTO.md`](./REPORTE_PRODUCTO.md)

---

## Qué hace y para quién

1. **Usuario:** el productor o la cooperativa de papa del Callejón de Huaylas y de las partes altas de las cordilleras Blanca y Negra (3,000–4,200 m).
2. **Decisión:** *“¿Protejo mi cultivo esta noche (riego, cobertura, humo) o no hace falta?”*. INTI-3K le da la mínima, la hora y las horas bajo 0 °C **en su celda de 3 km**, y le muestra lo que habría dicho el GFS de 27 km.
3. **Cómo:** baja el GFS actual de Open-Meteo, lo pasa por nuestro modelo de Kaggle (`EnsembleV8`) y emite la alerta en hora Perú, con boletín en español y quechua y voz del navegador.

*Usuarios secundarios (mismo motor, otras vistas):* COER Áncash y municipios (polígono por distrito), guías de montaña (ruta A→B con sensación térmica), brigadas (cono de humo con el viento 3 km).

---

## Instalación y ejecución en local (3 pasos)

Requisitos: **Python 3.12**, **Node.js 18+** (probado con Node 24), Git. No se necesita GPU ni ninguna clave de API.

**Paso 1 — Clonar e instalar dependencias**

<!-- TODO EQUIPO: reemplazar <URL_DEL_REPO> por la URL pública del repositorio en GitHub -->

```bash
git clone <URL_DEL_REPO> && cd <carpeta_del_repo>   # TODO EQUIPO: poner la URL pública real
pip install -r requirements.txt
cd product_app/frontend && npm install && cd ../..
```

**Paso 2 — Poner la data estática oficial en `data/`**

El producto solo necesita `data/static.npz` (relieve y coordenadas de las grillas). La data no se sube a GitHub. Bájala de la pestaña *Data* de la competencia en Kaggle, o con la API (requiere `kaggle.json` y haber aceptado las reglas):

```bash
kaggle competitions download -c hackathon-ia-pucp-2026
unzip hackathon-ia-pucp-2026.zip -d data/
```

Los pesos (`models/modelo.pt`, 23.11 MB) y `models/norm_stats.npz` ya vienen en el repositorio.

**Paso 3 — Levantar backend y frontend (dos terminales)**

```powershell
# Terminal 1 (Windows): verifica pesos + static.npz y levanta FastAPI en :8000
powershell -ExecutionPolicy Bypass -File .\start_local.ps1
```
```bash
# Terminal 1 (macOS/Linux), equivalente:
cd product_app/backend && python -m uvicorn app.main:app --port 8000
```
```bash
# Terminal 2: visor web
cd product_app/frontend && npm run dev
```

Abre **http://localhost:3000**. Comprobación rápida: `http://localhost:8000/api/health` responde JSON y `http://localhost:8000/docs` muestra la API.

> **Sin internet:** si Open-Meteo no responde, el backend usa la última descarga guardada (`product_app/backend/data/gfs_fallback_cache.npz`) y la interfaz lo indica con “SIN CONEXIÓN · usando caché GFS”. Los dos casos históricos de julio 2024 ya vienen en caché (`product_app/backend/data/gfs_hist/`).

<!-- TODO EQUIPO: docker-compose.yml no monta data/ y la imagen del backend no copia data/static.npz, así que hoy el contenedor responde 503. Agregar el volumen `./data:/workspace/data:ro` al servicio backend antes de anunciar `docker compose up --build` como opción. -->

**Alternativa mínima por línea de comandos** (el ejemplo oficial adaptado a nuestro modelo):

```bash
python app/ejemplo_producto.py --horas 24
python app/ejemplo_producto.py --desde 2024-07-18 --hasta 2024-07-18   # fecha pasada
```

### Cómo usarlo para la decisión de esta noche

1. Arriba se ve el estado del GFS (en vivo, desde caché o sin conexión) y la hora de descarga en hora Perú.
2. Elige **Pin** o **Polígono** y marca tu parcela en el mapa 3D.
3. El panel derecho muestra: mínima en tu celda 3 km y su hora, horas bajo 0 °C, y el recuadro **“Lo que el GFS 27 km no ve · tu celda 3 km”** (*“GFS 27 km decía X °C, INTI-3K dice Y °C”*).
4. Debajo aparece el boletín (se genera solo tras el análisis): cambia entre **Español** y **Quechua Áncash** y pulsa **Escuchar**.
5. Para ver un caso real, elige en la línea de tiempo **“Helada 18 jul 2024”** o **“Helada 29 jul 2024”**, o escribe una fecha pasada.

| Tecla | Acción |
| :---: | :--- |
| `1`–`4` | Vista: satélite · térmica (`T`) · viento 3 km · riesgo |
| `G` | Alternar **GFS 27 km** ↔ **INTI-3K 3 km** sobre el mapa |
| `Espacio` | Reproducir / pausar las horas (24 h o 48 h, hora Perú UTC−5) |
| `H` | Ocultar / mostrar paneles |
| `R` / `Esc` | Volver a la vista general |

---

## El modelo

<!-- ACTUALIZAR CON MODELO FINAL: toda esta tabla (nombre, scores, tamaño, épocas, tiempos) -->

| | |
|---|---|
| Modelo | **`EnsembleV8`**: 4 redes residuales (ConvNeXt, multi-escala, U-Net, FNO) con pesos de mezcla por variable |
| Entrada | GFS 17×13 con lags −24, −6, −3, −1, 0, +1, +3, +6, +24 h, media y desviación de 24 h, gradientes, divergencia, vorticidad, punto de rocío, hora UTC + 10 canales de relieve del DEM |
| Salida | `RH`, `T`, `U`, `V` en 70×35 celdas de 3 km |
| Skill en validación | **0.4100** <!-- ACTUALIZAR CON MODELO FINAL --> (holdout temporal 16–30 jun 2024). Por variable: RH 0.510 · T 0.587 · U 0.308 · V 0.235 |
| Leaderboard público Kaggle | **0.49844** <!-- ACTUALIZAR CON MODELO FINAL --> |
| Pesos | `models/modelo.pt`, **23.11 MB** (23,112,355 bytes, solo `state_dict`) · SHA-256 en `models/model_card.json` |
| Inferencia de 24 h | ~0.19 s en GPU (RTX 4050) · ~4 s en CPU de laptop |

**Sobre el 0.4100:** es optimista. La época de parada de cada red se eligió mirando esa misma validación (16–30 jun), así que no es una estimación independiente. El LB público (0.49844) usa otro período (julio–agosto) y no es comparable 1 a 1. El ranking que cuenta es el privado.

### Reproducir el modelo y el `submission.csv`

El notebook **`train/submission_notebook.ipynb`** usa solo `data/train.npz`, `data/test.npz` y `data/static.npz`, con `SEED = 42`:

- **`TRAIN = True`** (valor por defecto): Fase A (entrena las 4 redes con corte temporal, val = 16–30 jun, y elige las épocas), Fase B (reentrena con el 100 % de febrero–junio), escribe `models/modelo.pt`, `models/norm_stats.npz` y `submission.csv` (646,800 filas, sin NaN, `RH` en [0, 100]). Tiempo: **~35–40 min en GPU** <!-- ACTUALIZAR CON MODELO FINAL --> (salidas guardadas: Fase A ≈ 21 min, Fase B ≈ 15 min, en CUDA).
- **`TRAIN = False`**: no entrena; carga los pesos de `models/` y solo genera el `submission.csv`.

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 train/submission_notebook.ipynb
```

También corre en Kaggle (GPU T4) o Colab.

### Validación temporal (nunca split aleatorio)

`kaggle_pipeline/src/cv.py` define 3 folds *rolling-origin* con 24 h de separación entre train y val: 16–31 may, 1–15 jun y 16–30 jun (pesos 0.2 / 0.3 / 0.5). La climatología, la normalización y los pesos de la pérdida se calculan solo con el train de cada fold. Se corre por red con `python kaggle_pipeline/src/run_cv.py --model unet` (opciones con `--help`). Los resultados quedan en `kaggle_pipeline/experiments/registro.csv`.
<!-- ACTUALIZAR CON MODELO FINAL: hoy el registro solo tiene el fold 3 (0.4100) de EnsembleV8; completar folds 1 y 2 y la cv_media -->

### Tests

```bash
python -m pytest tests -q                       # métrica, folds sin fuga, mezcla, formato del submission, paridad
python -m pytest product_app/backend/tests -q   # lags ±24 h idénticos al entrenamiento, caché offline, exposición a helada
```

Última corrida local: 31 + 11 tests pasan (~1 min en CPU).

---

## Estructura del repositorio

```text
├── README.md · REPORTE_PRODUCTO.md · requirements.txt · start_local.ps1 · docker-compose.yml
├── data/                       ← data oficial de Kaggle (no se sube; ver Paso 2)
├── train/
│   ├── starter_notebook.ipynb  ← baseline oficial
│   └── submission_notebook.ipynb ← notebook final reproducible (flag TRAIN)
├── models/
│   ├── modelo.pt               ← pesos de EnsembleV8 (23.11 MB), los mismos del producto
│   ├── norm_stats.npz          ← estadísticas de normalización
│   └── model_card.json         ← scores, pesos de mezcla, SHA-256
├── app/
│   ├── gfs_forecast.py         ← descarga del GFS de Open-Meteo (oficial)
│   ├── inti_inference.py       ← única entrada del producto al modelo (reusa el preprocesamiento de entrenamiento)
│   └── ejemplo_producto.py     ← ejemplo CLI de punta a punta
├── kaggle_pipeline/src/        ← entrenamiento (train_v7/v8), CV temporal (cv.py, run_cv.py), métrica
├── tests/                      ← tests del pipeline del modelo
└── product_app/
    ├── backend/                ← FastAPI: caché GFS, inferencia, comparador, boletín
    └── frontend/               ← Next.js + MapLibre (mapa 3D sin claves)
```

Variables opcionales en `.env.example` (`FIRMS_MAP_KEY` para focos de incendio reales, `OLLAMA_URL` para un LLM local). Sin ellas el producto funciona completo.
