# PROMPT DE RELEVO — Equipo PASSWORD · Hackathon IA PUCP 2026

> Copia TODO lo que está debajo de la línea y pégalo como primer mensaje a Claude Code
> (o a tu agente) en la otra laptop, abierto en la carpeta del repo clonado.

---

Eres el líder técnico del equipo **PASSWORD** en la Hackathon IA PUCP 2026 (Kaggle `hackathon-ia-pucp-2026`).
Objetivo: **subir al puesto 1 del leaderboard privado** y **entregar todo lo obligatorio antes del martes 8 dic 23:59 (hora Lima)**, sin romper nada que ya funciona. Trabaja con autonomía, usa subagentes especializados en paralelo cuando las tareas no se pisen, y verifica todo con números antes de afirmar.

## 0. Arranque (haz esto primero)
1. Repo: `https://github.com/DanielAlmeida023/PASSWORD_IAPUCP`, rama **`plan-ataque`** (`git fetch && git checkout plan-ataque`). Lee completos: `README.md`, `HANDOFF_PROMPT.md` (este archivo), `kaggle_pipeline/experiments/registro.csv`, y los docstrings de `kaggle_pipeline/src/{run_cv,cv,metrics,blend,ensemble_vn,export_ensemble}.py`, `kaggle_runner/{launch,scheduler}.py`, `app/inti_inference.py`.
2. Data oficial: `pip install kaggle`, token en `~/.kaggle/kaggle.json` (lo pone el humano, nunca lo escribas tú), aceptar reglas en la web, `kaggle competitions download -c hackathon-ia-pucp-2026 -p data && unzip` → `data/{train,test,static}.npz, sample_submission.csv`.
3. Artefactos de experimentos (no están en git): el equipo te pasa `artefactos_password.zip` (Drive/WhatsApp). Descomprímelo en la raíz del repo; crea `kaggle_pipeline/experiments/oof/` (predicciones OOF por miembro y fold), `kaggle_pipeline/experiments/refit/` (modelos refit 100 % + pred_test) y `models_candidates/vn4/`. Si no llega, se regeneran con los comandos de §4 (≈ 15–25 min por job en Kaggle T4).
4. Entorno: Python 3.12, `pip install -r requirements.txt` (torch con CUDA si tienes GPU NVIDIA usable). Corre `python -m pytest tests -q` (43 tests) y `cd product_app/backend && python -m pytest tests -q` (14 tests). Todo debe pasar antes de tocar nada.
5. **Entrena en Kaggle GPU (T4), no confíes en GPUs locales inestables.** Con tu propia cuenta Kaggle (verificada por teléfono): `python kaggle_runner/launch.py sync` sube `kaggle_pipeline/src` como dataset PRIVADO `<tu_usuario>/password-src`; `python kaggle_runner/scheduler.py kaggle_runner/jobs.txt` mantiene 2 kernels GPU privados en cola y baja OOF/refits solos a `kaggle_pipeline/experiments/`. Cada cuenta del equipo tiene ~30 h GPU/semana: si hay varias laptops, repartan los jobs (cada cuenta su propia cola) y junten los artefactos. En Windows exporta `PYTHONIOENCODING=utf-8`.

## 1. Reglas duras (no negociables)
- Solo data oficial (`train.npz`, `test.npz`, `static.npz`) para entrenar y predecir. Nunca split aleatorio; nunca usar el test para elegir hiperparámetros.
- **El producto debe usar EXACTAMENTE los mismos pesos que la submission final** (descalificación si no). Tras cada cambio de modelo final: `models/modelo.pt` + `models/norm_stats.npz` + `models/model_card.json` vía `export_ensemble.py`, y re-correr `tests/test_paridad.py` (max|diff| < 1e-3).
- `train/submission_notebook_vn.ipynb` debe, corrido de principio a fin solo con data oficial, entrenar el ensamble final y generar exactamente el `submission.csv` final, **guardado con salidas**. Debe caber en una sesión Kaggle (12 h) o documentar `TRAIN=False`.
- Pesos < 100 MB en git (si no, link en README). Ninguna clave/API key en el repo.
- **No envíes a Kaggle ni hagas `git push` sin OK explícito del humano.** Deja los CSV en `submissions/` con nombre `sub_{fecha}_{id_exp}_cv{cv:.4f}.csv`, validados con `kaggle_pipeline/src/check_submission.py`. Registra TODO experimento (también los descartados) en `kaggle_pipeline/experiments/registro.csv`.
- Una mejora se queda solo si sube `cv_media` (ponderada 0.2/0.3/0.5 de los folds) ≥ +0.002 de forma consistente. Elegir finales por CV, nunca solo por LB público (≈ 79 horas, ruidoso).

## 2. Contexto técnico mínimo
- Tarea: GFS 27 km `X (17×13×4)` (fila 0 = norte) → WRF 3 km `Y (70×35×4)` (fila 0 = sur) + DEM; variables `RH %, T K, U, V m/s`. Métrica `1 − mean_v(RMSE_v / RMSE_clim_v)`, clim por hora UTC. Kaggle evalúa **solo horas 00,03,…,21 UTC** del test (264 h × 2450 celdas = 646 800 filas). Público ≈ 30 %, privado ≈ 70 %.
- Modelo: miembros DirectRes (UNet, ConvNeXt, MultiScale, FNO) de `train_v8.py` con TemporalLagMixer sobre lags GFS `(-24,-6,-3,-1,0,1,3,6,24)`, ancla física (lapse rate aprendible por hora) y pérdida alineada con la métrica. `run_cv.py` entrena 1 miembro (CV 3 folds o `--refit`). `EnsembleVN` combina N miembros con pesos por variable.
- CV (`cv.py`): fold1 val 16–31 may, fold2 1–15 jun, fold3 16–30 jun; train = todo lo anterior con gap de 24 h. `metrics.kaggle_like_score` replica Kaggle.
- **Receta de refit validada:** calendario OneCycle de 24 épocas y parar en la época validada: `--refit --epochs 24 --stop-epoch 12` (UNet/FNO), `--stop-epoch 16` (ConvNeXt/MultiScale). Comprobado: CV sin selección por val con stop 12 = 0.3877 vs 0.3878 con selección. **No** uses calendarios cortos (`--epochs 12` completo = 0.3752, −0.013).
- Lags futuros (t+1..t+24): las bases solo describen la entrada t−2..t. **Pendiente de confirmar en el grupo de participantes.** Plan B listo: `--lags past` (lags causales `(-24,-6,-3,-1,0,-2,-12,-9,-18)`, ventana M24/S24 causal). Si la organización lo prohíbe, TODO el ensamble final debe re-entrenarse con `--lags past`.

## 3. Estado al 26 nov 2026, 15:20 (Lima)
LB público: 1.º SciTeX 0.51938 · 2.º 0.51030 · 3.º 0.50912 · 4.º 0.50570 · **PASSWORD ≈ 0.50029** (mejor envío). Brecha al 1.º ≈ +0.019.

| Experimento | cv_media | LB público | Nota |
|---|---|---|---|
| UNet s42 | 0.3878 | — | mejor miembro (U 0.265, V 0.191) |
| UNet s203 | 0.3854 | — | |
| ConvNeXt s42 | 0.3849 | — | |
| MultiScale s777 / s102 | 0.3802 / 0.3797 | — | |
| FNO s304 (modes 12×8) | 0.3740 | — | débil solo, **pero suma en el blend** |
| FNO compacto 8×6 wd 4e-2 | 0.3685 | — | DESCARTADO |
| UNet loss U,V ×1.3 | 0.3854 | — | DESCARTADO |
| UNet pesos por recencia "plan" | 0.3864 | — | DESCARTADO (pero fold3 0.3936 ↑) |
| Blend 4 (cnx42, ms102, ms777, unet42) uniforme | 0.4060 | **0.50029** | envío actual |
| Blend 6 (+unet203, +fno304) uniforme | **0.4091** | pendiente de enviar | **listo**: `submissions/sub_20261126_vn6_uniform_cv0.4091.csv`, modelo en `models_candidates/vn6/` (36 MB) |
| EnsembleV8 viejo (blend uniforme) | — | 0.49976 | calibración |
| EnsembleV8 viejo (blend optimizado) | 0.4100* | 0.49844 | *optimista (época elegida en val) |
| EnsembleV8 sin FNO | — | 0.49464 | el FNO aporta |

Aprendizajes clave: (1) **el ensamble es la palanca principal** (+0.018 de 1→4 miembros, +0.003 más con 6); cada miembro diverso suma. (2) Optimizar pesos del blend casi no aporta (+0.0003) y en LB el uniforme ganó al optimizado → usar uniforme o NNLS con shrinkage fuerte. (3) El viento (U, V skill 0.19–0.26) es la variable más débil y la más barata de subir. (4) CV (~0.40) y LB (~0.50) tienen escalas distintas (la clim de referencia difiere); compara órdenes, no valores.

Cola Kaggle que quedó pendiente (`kaggle_runner/jobs.txt`, el scheduler la reanuda con `kaggle_runner/outputs/scheduler_state.json`): `r2-fno-s304` (refit), `cv-unet-lagpast-s42` (plan B), `cv-unet-s777`, `cv-unet-s1234`, `cv-convnext-s777`. Si esos resultados no llegaron en los artefactos, relánzalos desde tu cuenta.

Mejoras de viento YA implementadas en la rama (apagadas por defecto, default bit-idéntico, 58+14 tests OK): flags de `run_cv.py` `--topo-extra` (9 canales del DEM: TPI k=21, curvatura k=3/9, aspecto sin/cos, pendiente a lo largo/transversal al eje del valle estimado por PCA del DEM ≈ 334°, distancia al eje) y `--wind-head` (cabeza U/V con 2 bloques residuales dilatados). Export/producto ya las reconstruyen (`members_config`). **Aún NO medidas en GPU.** Estudio offline (`kaggle_pipeline/src/wind_study.py`, resultados en `experiments/wind_study.json`): el error de U/V no es sesgo (|sesgo| ≤ 0.12 m/s) sino predicción **demasiado suave** (std pred 2.2–2.6 vs real 2.5–2.9), peor en fondos de valle (decil TPI más bajo: skill 0.13–0.18) y de 03z a 12z (noche local). Las correcciones post-hoc (sesgo por hora, a·y+b) **no pasan** el umbral (máx +0.0006) → descartadas. Ideas que atacan la suavidad: pérdida con término de gradiente espacial o de varianza para U/V, cabeza de viento, features de valle.

## 4. Plan para llegar al puesto 1 (en orden de ganancia esperada / costo)
1. **Envío del blend de 6** (cv 0.4091): el CSV ya está generado (pedir OK y subir). Comando de referencia por si hay que regenerarlo:
   `python kaggle_pipeline/src/export_ensemble.py --members convnext_42 multiscale_102 multiscale_777 unet_42 unet_203 fno_304 --blend uniform --refit-dir kaggle_pipeline/experiments/refit --out models_candidates/vn6 --sub-dir submissions --id-exp vn6_uniform --cv-media 0.4091 --device cpu` → pedir OK → `kaggle competitions submit ...` → registrar LB.
2. **Más semillas y diversidad** (casi seguro +0.003 a +0.008): CV + refit de UNet s777/s1234, ConvNeXt s777/s1234, MultiScale s42, FNO s42/s777 (modes 12×8 original). Medir siempre con `blend.py --members ... --shrink 0.3 1.0` (mira la línea NESTED/uniforme). Parar cuando agregar miembros suba < +0.001. Vigilar tamaño: 10–12 miembros ≈ 60 MB fp32 (export guarda fp16 si > 95 MB).
3. **Viento (la plata para el 1.º):** primero lanzar las 4 líneas `cv-unet-topox-*`/`cv-unet-wh-*` que ya están en `jobs.txt` y comparar contra `unet_42` (0.3878) **y** `unet_203` (0.3854) — exigir mejora en ambas semillas. Luego medir contra UNet s42 (misma semilla, 24 épocas) cada opción: features topográficas extra, cabeza de viento, y corrección de sesgo/amplitud por hora UTC×variable estimada anidada en OOF (solo escalares, nunca por píxel). Quedarse con lo que suba cv_media ≥ +0.002 en los 3 folds. Ideas extra si sobra tiempo: pesos por recencia suaves (fold3 sugiere que ayudan al periodo seco del test), SWA en el último 25 % de épocas, `noise_std ∈ {0.03, 0.06}`, stacker CNN pequeño sobre OOF (solo si +0.004).
4. **Plan B listo para sustituir** si los lags futuros están prohibidos: CV de cada arquitectura con `--lags past` y refits; mismo flujo.
5. Presupuesto de envíos: 5/día (por equipo). Usa 2–3/día en candidatos reales; no gastes en variaciones mínimas.

## 5. Elección de finales y cierre (antes del 7 dic)
- Final 1: **mejor `cv_media`**. Final 2: el ensamble más grande/diverso con CV ≥ mejor − 0.002 (cobertura de riesgo). Nunca uno que suba en LB pero baje en CV.
- Con el final 1: `export_ensemble.py ... --out models/` (sobrescribe los pesos del producto), actualiza `models/model_card.json` (cv, LB), corre `tests/test_paridad.py`, levanta el producto (`start_local.ps1` o según README) y verifica que `/api/model-info` muestre el sha correcto.
- Pon la lista final en `MEMBERS`/`BLEND` de `train/submission_notebook_vn.ipynb`, córrelo completo en Kaggle GPU (subiendo el repo como dataset o `git clone`), guárdalo **con salidas**, y verifica que su `submission.csv` coincida con el final enviado (max|diff| ≲ 1e-3; si no coincide por no-determinismo GPU, documéntalo y envía el CSV que produce el notebook).
- Reemplaza `train/submission_notebook.ipynb` por el notebook final (o deja claro en el README cuál es el oficial). Fija versiones exactas en `requirements.txt`.
- README: actualiza todo lo marcado `<!-- ACTUALIZAR CON MODELO FINAL -->`. El equipo llena `<!-- TODO EQUIPO -->` (integrantes, link del video, URL pública). REPORTE_PRODUCTO.md §1 con números finales.
- Prueba en limpio: clonar en otra carpeta, seguir SOLO el README, levantar el producto, generar un pronóstico con GFS actual y uno de fecha pasada (helada 18 jul 2024). Corregir cualquier paso faltante.
- Repo público, video ≤ 5 min público (guion en `docs_entrega/GUION_VIDEO.md`: usar el caso **18 jul 2024, chacra a 4,000 m: "GFS decía 4.6 °C, INTI-3K dice −0.8 °C"**), nombre del equipo en Kaggle = `PASSWORD`, formulario de entregables enviado el 7 dic.

## 6. Producto INTI-3K (ya hecho; solo no romperlo)
Backend FastAPI (`product_app/backend/app/main.py`) + Next.js (`product_app/frontend`). Usuario principal: productor de papa del Callejón de Huaylas, decisión "¿protejo mi chacra esta noche por helada?". Ya incluye: lags ±24 h correctos (pide 24 h pasado y horizonte+24 futuro), caché offline de Open-Meteo con refresco 3 h, fechas pasadas (`/api/historical-forecast`), comparador GFS vs INTI-3K por celda 3 km, semáforo basado solo en helada (incendio/viento como alertas secundarias), horas en hora Perú (UTC−5), badge dinámico desde `/api/model-info`, boletín en quechua por plantillas (sin LLM). Docker: `docker-compose.yml` monta `./data` read-only (build no probado aún).

## 7. Forma de trabajar
- Primero lee, después actúa. Antes de cada envío o push, pide OK con: archivo, cv_media, qué cambió.
- Paraleliza con subagentes: (a) cola de Kaggle y blends, (b) viento/features, (c) notebook final + paridad + README/prueba en limpio. Nunca dos agentes editando los mismos archivos.
- Reporta al humano en español, corto, con tablas de números (cv por fold, cv_media, LB) y qué decides.
