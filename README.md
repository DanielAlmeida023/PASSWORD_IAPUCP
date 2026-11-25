# INTI-3K · Centro de Comando Microclimático 3 km (Cordillera Blanca)

**Hackathon IA PUCP 2026 (DSC PUCP × IA-PUCP) · Equipo `PASSWORD`**

**INTI-3K** transforma el pronóstico global borroso del clima (**GFS ~27 km**) en mapas de alta resolución (**3 km**, grilla `70×35`) de humedad relativa (`RH`), temperatura (`T`) y viento (`U, V`) sobre el relieve de la Cordillera Blanca y el Callejón de Huaylas, mediante un **ensamble unificado guiado por física andina** (`models/modelo.pt`) y un **visor táctico 3D *100% Keyless*** orientado al COER Áncash, municipalidades, SERNANP y comunidades agrarias.

- **Video Pitch (máx. 5 min):** `[Pegar aquí el link público al Video Pitch]`
- **Integrantes (Equipo `PASSWORD`):** `[Integrante 1]`, `[Integrante 2]`, `[Integrante 3]`, `[Integrante 4]`
- **Detalle técnico y propuesta de valor:** Ver [`REPORTE_PRODUCTO.md`](./REPORTE_PRODUCTO.md)

---

## Qué hace y para quién

* **Usuario concreto:** Operadores del **COER Áncash**, municipios altoandinos (Huaraz, Caraz, Chacas, Cátac/Recuay), guardaparques (**SERNANP / Huascarán**) y cooperativas agrarias.
* **Capacidades clave del producto (`app/` y `product_app/`):**
  * **Visor 3D Táctico *Keyless*:** Terreno 3D satelital (Esri World Imagery + AWS Terrarium DEM, **sin API keys**) con 4 ópticas en tiempo real: Satélite, **FLIR Térmico (`T`)**, **Viento Encajonado (`U, V`)** y **Riesgo Compuesto (Heladas + Incendios)**.
  * **Comparador GFS 27 km vs. INTI-3K 3 km (`tecla G`):** Muestra visualmente las diferencias térmicas de hasta `14–30 °C` entre valles y nevados que el pronóstico global omite.
  * **Dibujo libre (Pin, Polígono de Parcela/Distrito, Ruta A→B):** Evalúa al instante área afectada (`km²`), horas bajo `0 °C`, *Wind Chill* y perfil de elevación sobre las celdas exactas de 3 km.
  * **Cono de humo e incendios 3 km:** Proyecta la dispersión de humo durante 8 horas usando los vectores `(U, V)` y `RH` de `models/modelo.pt` (con integración opcional a **NASA FIRMS** en vivo o simulador *keyless*).
  * **Boletín bilingüe (Español + Quechua Áncash) + Voz nativa:** Genera alertas accionables listas para leer o escuchar sin servicios de pago (con soporte opcional a **Ollama** local).

---

## Instalación y Ejecución del Producto en Local (3 pasos)

### 1. Descarga la data estática oficial en `data/`
*(Solo necesaria si aún no tienes `data/static.npz`)*:
```bash
kaggle competitions download -c hackathon-ia-pucp-2026
unzip hackathon-ia-pucp-2026.zip -d data/
```
*(El modelo ya entrenado `models/modelo.pt` de `30.70 MB` y las estadísticas `models/norm_stats.npz` ya vienen incluidos en el repositorio).*

### 2. Opción A — Arranque completo con Docker Compose (1 comando)
```bash
docker compose up --build
```
- **Producto Web 3D (Frontend):** `http://localhost:3000`
- **Backend API & Swagger Docs:** `http://localhost:8000/docs`

### 2. Opción B — Arranque directo en local (Python + Node.js)
1. **Instalar dependencias de Python y levantar el Backend FastAPI:**
   ```bash
   pip install -r requirements.txt
   cd product_app/backend
   uvicorn app.main:app --reload --port 8000
   ```
2. **En otra terminal, levantar el Frontend 3D (`Next.js`):**
   ```bash
   cd product_app/frontend
   npm install
   npm run dev
   ```
   Abre **`http://localhost:3000`** en tu navegador.

*(También puedes ejecutar el script CLI mínimo oficial de punta a punta desde la raíz)*:
```bash
python app/ejemplo_producto.py --horas 24
```

### Atajos de Teclado en el Visor 3D (Demo)
| Tecla | Acción |
| :---: | :--- |
| `1` – `4` | Cambiar óptica: `[1]` Satélite · `[2]` FLIR Térmico · `[3]` Viento 3 km · `[4]` Riesgo |
| `G` | Alternar comparador **GFS 27 km** vs. **INTI-3K IA 3 km** |
| `Espacio` | Play / Pause del reproductor temporal (24h–48h en Hora Perú UTC−5) |
| `H` | Ocultar / Mostrar paneles HUD |
| `R` / `Esc` | Reiniciar vista panorámica 3D de la Cordillera Blanca |

---

## Reproducibilidad del Modelo en Kaggle (`train/submission_notebook.ipynb`)

El notebook **`train/submission_notebook.ipynb`** es **100% autocontenido** (compatible con Local, Google Colab y Kaggle GPU T4), fija todas las semillas (`SEED = 42`) y, ejecutado de principio a fin usando **exclusivamente** `data/train.npz`, `data/test.npz` y `data/static.npz`:
1. Entrena las 3 familias complementarias (`PhysicsUNet`, `PhysicsConvNeXt` y `PhysicsFNO`) con validación temporal estricta (`ttr >= '2024061600'`).
2. Construye el modelo único **`EnsembleDownscaler`** (**Skill Score local: `0.3821`** vs. `0.3357` del starter).
3. Exporta los pesos limpios a `models/modelo.pt` (`30.70 MB`) y `models/norm_stats.npz`.
4. Genera y verifica `submission.csv` (`646,800` filas, `0` NaNs, `RH` en `[0, 100]`).

Para re-ejecutarlo desde terminal:
```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 train/submission_notebook.ipynb
```

---

## Estructura del Repositorio

```text
├── README.md                              ← Guía de instalación, uso y reproducibilidad
├── REPORTE_PRODUCTO.md                    ← Reporte oficial de 6 secciones para el jurado
├── requirements.txt                       ← Dependencias de Python
├── .env.example                           ← Variables opcionales (FIRMS_MAP_KEY, OLLAMA_URL)
├── train/
│   ├── starter_notebook.ipynb             ← Baseline oficial
│   └── submission_notebook.ipynb          ← Notebook final autocontenido con salidas guardadas
├── models/
│   ├── modelo.pt                          ← Pesos limpios del EnsembleDownscaler (30.70 MB)
│   └── norm_stats.npz                     ← Estadísticas de normalización del split de entrenamiento
├── app/
│   ├── gfs_forecast.py                    ← Conector oficial Open-Meteo GFS (con caché)
│   └── ejemplo_producto.py                ← Inferencia oficial de punta a punta con models/modelo.pt
└── product_app/
    ├── backend/                           ← API FastAPI, GeoJSON andino y caché offline de respaldo
    └── frontend/                          ← Visor 3D táctico en Next.js + MapLibre GL + Tailwind
```
