# Reporte del producto · INTI-3K (Equipo PASSWORD)

**Integrantes:** [Integrante 1], [Integrante 2], [Integrante 3], [Integrante 4]  
**Producto:** Se ejecuta en local en 2 pasos o mediante `docker compose up --build` (ver `README.md`)  
**Video pitch:** [Pegar link público del video pitch ≤ 5 min]

---

## 1. ¿Qué modelo o arquitectura usaron?

Construimos **`EnsembleV8`**, un modelo único unificado en PyTorch (`models/modelo.pt`) que integra **cuatro familias arquitectónicas *Full DirectRes* complementarias** con atención temporal multi-cabeza **`TemporalLagMixer`** (alinea las 9 ventanas diurnas `±24h` por celda GFS) y atención espacial **`GFS2WRFCrossAttention`** (permite que cada celda regional de 3 km consulte celdas GFS de barlovento/sotavento) sobre media y desviación estándar móvil de 24h (`M24`, `S24`) y **gradientes espaciales GFS** ($\nabla T$, $\nabla RH$, **divergencia** $\partial_x U + \partial_y V$ y **vorticidad** $\partial_x V - \partial_y U$):

1. **`DirectResConvNeXtV8`** (`w=84`, kernels `7×7` dilatados + *Cross-Attention* + `DirectPhysicalAnchor`): captura la estructura fina del relieve (`0.3809`–`0.3879` individual).
2. **`DirectResMultiScaleV8`** (`w=96`, bloques dual-escala paralelos `3×3 + 7×7` dilatados + `DirectPhysicalAnchor`): combina microescala de quebradas y flujo regional (`0.3828` individual).
3. **`DirectResUNetV8`** (`w=64`, U-Net 3 niveles + *Cross-Attention* en `18×9` + `DirectPhysicalAnchor`): modela circulación de valle multiescala sin arrastre de climatología húmeda (`0.3903` individual, líder en viento `U=0.2986, V=0.2224`).
4. **`DirectResFNOV8`** (*Fourier Neural Operator 2D* compacto regularizado `w=32`, modos `12×8` + `DirectPhysicalAnchor`): captura ondas atmosféricas amplias y humedad regional (`0.3744` individual).

Todas las ramas eliminan por completo la dependencia de la climatología húmeda de febrero–abril usando **`DirectPhysicalAnchor`** (ancla directa sobre GFS interpolado + gradiente térmico altitudinal `lapse_rate_h` y pendiente de humedad `rh_elev_h` aprendibles por hora UTC + sesgo diurno 3 km + prior adiabático seguro de `RH`), comparten **10 canales topográficos** del `DEM`, punto de rocío (Magnus) y forzamiento orográfico ($\vec{V}_{\text{GFS}} \cdot \nabla \text{DEM}$), y optimizan directamente el **Skill Score oficial** con ponderación estacional y *EMA*.

**Protocolo en 2 fases (validación estricta + máximo desempeño en julio–agosto):**
* **Fase A (Validación Temporal):** Corte causal estricto (`val = ttr >= '2024061600'`, 336 horas) para medir el Skill Score real (`0.4100` en notebook; `0.4071` en validación `train_v8.py`) y calibrar los pesos en el simplex y las épocas óptimas.
* **Fase B (Refit Final 100% Data Oficial):** Reentrenamiento final de las 4 familias sobre el **100% de `train.npz` (`2,935` horas: 18 feb – 30 jun)** usando las épocas óptimas validadas (`[16, 19, 14, 14]`), incorporando las últimas dos semanas secas de junio antes de predecir julio–agosto (`test.npz`).

| Métrica / Especificación | Valor Medido |
| :--- | :--- |
| Arquitectura | `EnsembleV8` (`DirectResConvNeXtV8` + `DirectResMultiScaleV8` + `DirectResUNetV8` + `DirectResFNOV8`) |
| Parámetros / tamaño de los pesos | `5,594,404` parámetros · **`23.11 MB`** (`models/modelo.pt`, solo `state_dict`) |
| Score en el leaderboard público | **`0.49844`** (puesto #4 en Public LB · primer envío `EnsembleV8`) |
| Score en validación local (`>= 2024061600`) | **`0.4100`** (vs. `0.3357` del starter y `0.3984` de v7) · Skill: `RH 0.5097`, `T 0.5865`, `U 0.3082`, `V 0.2354` · RMSE: `RH 8.42%`, `T 0.83 K`, `U 1.56 m/s`, `V 1.49 m/s` |
| Tiempo de inferencia (24 horas de pronóstico) y hardware | **`185.9 ms`** en NVIDIA GeForce RTX 4050 (6 GB VRAM) |

---

## 2. ¿Para quién es y qué problema le resuelve?

**INTI-3K** está diseñado para **operadores del COER Áncash, municipalidades altoandinas (Huaraz, Caraz, Chacas, Cátac/Recuay), guardaparques del SERNANP (Parque Nacional Huascarán) y cooperativas agrarias del Callejón de Huaylas**.

Hoy, estas instituciones dependen de pronósticos globales como el GFS (~27 km por celda), que mezclan en un solo número valles agrícolas a 2,200 m con nevados a 5,700 m. Como resultado, un agricultor no sabe si caerá helada en su quebrada específica, ni un brigadista sabe hacia qué centro poblado empujará el viento local el humo de un incendio de pastizales o qué ventana horaria es segura en rutas de alta montaña. Con INTI-3K, pasan de reaccionar ante emergencias con información borrosa a tomar decisiones preventivas a **resolución de 3 km (`70×35` celdas)** antes del anochecer.

---

## 3. ¿Cuál es la propuesta de valor del producto?

INTI-3K convierte en **menos de 0.2 segundos** un pronóstico global borroso de 27 km en inteligencia operativa a **3 km** sobre el relieve 3D real de la Cordillera Blanca, revelando diferencias térmicas valle-cumbre de hasta **14–30 °C** que el GFS oculta. Esto permite proteger cultivos antes de heladas nocturnas, anticipar la dirección del humo/fuego con horas de ventaja y comunicar la alerta en **Español y Quechua Áncash** mediante boletines listos para radio comunitaria.

---

## 4. ¿Por qué tu producto es diferente a algo usual?

A diferencia de una aplicación de clima convencional o un mapa 2D estático, INTI-3K integra cinco capacidades operativas construidas sobre la grilla de 3 km:

1. **Comparador directo `GFS 27 km vs. INTI-3K 3 km` en 3D:** El usuario alterna en un clic (`tecla G`) sobre terreno satelital 3D para comprobar visualmente cuánto error habría cometido usando el pronóstico global sin downscaling.
2. **Análisis geográfico libre (Pin, Polígono de Parcela/Distrito y Ruta A→B):** Cualquier usuario dibuja su propia parcela agrícola o ruta de rescate en el mapa y obtiene el cálculo exacto sobre las celdas de 3 km contenidas, incluyendo horas bajo `0 °C` y sensación térmica (*Wind Chill*).
3. **Simulador de cono de humo y propagación de incendios 3 km:** Cruza focos activos (NASA FIRMS vía `.env` opcional o simulador integrado *keyless*) con la humedad (`RH`) y los vectores de viento `(U, V)` de nuestro modelo para proyectar hacia dónde avanzará la pluma de humo en las próximas 8 horas.
4. **Inclusión lingüística (Español + Quechua Áncash) y control por voz gratuito:** Traduce automáticamente las alertas al Quechua del Callejón de Huaylas y permite navegar o escuchar el boletín usando APIs nativas del navegador (y soporte opcional a Ollama local), sin requerir claves de pago.
5. **Rigor científico transparente y arranque 100% *Keyless*:** Muestra en pantalla el error real de validación de cada variable (`T ±0.83 °C`, `RH ±8.42%`, viento `±1.49–1.56 m/s`) y corre completo sin exigir cuentas ni tokens de terceros.

---

## 5. ¿Cómo usa el producto la salida de tu modelo?

El flujo opera de punta a punta sin atajos ni datos de relleno:

1. **Ingesta actual:** `app/gfs_forecast.py` descarga las últimas 24 horas pasadas (`past_hours=24`) y el pronóstico GFS (24h a 48h, o fechas pasadas) desde **Open-Meteo**, guardando caché en disco (`2h`) y un respaldo offline automático para garantizar disponibilidad.
2. **Inferencia 3 km:** El backend FastAPI ejecuta **exactamente los mismos pesos (`models/modelo.pt`)** generados por `train/submission_notebook.ipynb` (vía `app/ejemplo_producto.py`), produciendo los mapas `70×35` horarios de `RH`, `T`, `U` y `V`.
3. **Cálculo de decisiones:** A partir de esas matrices de 3 km, el backend calcula máscaras de helada (`T < 0 °C`), índice de peligro de incendio, *Wind Chill*, estadísticas por polígono/ruta y trayectorias de humo advectadas por `(U, V)`, que luego se renderizan en tiempo real sobre el visor 3D (`MapLibre GL`) y en el boletín bilingüe Español/Quechua Áncash.

---

## 6. Limitaciones y próximos pasos

* **Limitaciones actuales:** (a) El error residual en vientos encajonados de valles (`RMSE ~1.49–1.56 m/s`) sigue siendo el componente más complejo por la turbulencia sub-grilla menor a 3 km; (b) aunque la arquitectura *Full DirectRes* de `EnsembleV8` y el reentrenamiento con el 100% de la data mitigan la deriva hacia julio–agosto, el dataset oficial cubre 5 meses (febrero–junio); (c) en modo en vivo depende de Open-Meteo, mitigado mediante caché persistente y respaldo local.
* **Próximos pasos:** Entrenar sobre un ciclo anual completo (12 meses de simulaciones WRF), integrar estaciones meteorológicas automáticas del SENAMHI para calibración en tiempo real (*bias correction*) y conectar la exportación directa de boletines en audio a emisoras locales y WhatsApp comunitario del COER Áncash.
