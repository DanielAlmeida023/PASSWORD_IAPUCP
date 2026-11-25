# Reporte del producto · INTI-3K (Equipo PASSWORD)

**Integrantes:** [Integrante 1], [Integrante 2], [Integrante 3], [Integrante 4]  
**Producto:** Se ejecuta en local en 2 pasos o mediante `docker compose up --build` (ver `README.md`)  
**Video pitch:** [Pegar link público del video pitch ≤ 5 min]

---

## 1. ¿Qué modelo o arquitectura usaron?

Construimos **`EnsembleDownscaler`**, un modelo único unificado en PyTorch (`models/modelo.pt`) que combina tres familias arquitectónicas complementarias guiadas por física andina sobre ventanas temporales centradas del pronóstico global GFS (`[t-6, t-3, t-1, t, t+1, t+3, t+6]` horas) y su promedio móvil de 24 horas:

1. **`PhysicsUNet`** (pirámide multi-escala `3×3` con atención *Squeeze-and-Excitation*): especialista en la dinámica local del viento encajonado (`U`, `V`).
2. **`PhysicsConvNeXt`** (convoluciones con kernels grandes `7×7` y dilatadas a resolución completa `70×35` sin compresión espacial): especialista en el gradiente térmico sobre el relieve (`T`, `RMSE = 0.93 °C`).
3. **`PhysicsFNO`** (*Fourier Neural Operator 2D*): procesa frecuencias espaciales globales para capturar campos atmosféricos amplios y humedad relativa (`RH`).

Todas las ramas comparten **10 canales topográficos físicos** derivados del `DEM` oficial (elevación, gradientes Norte-Sur y Este-Oeste, pendiente, orientación de ladera y posición topográfica local/mesoescalar), el **punto de rocío** (fórmula de Magnus), un prior térmico por gradiente altitudinal (`-6.5 °C/km`) y el **forzamiento orográfico** ($\vec{V}_{\text{GFS}} \cdot \nabla \text{DEM}$). Entrenamos optimizando directamente la fórmula del **Skill Score oficial** con ponderación estacional (dando mayor peso a mayo–junio frente a febrero) y promedio móvil exponencial de pesos (*EMA*). La validación se realizó mediante **corte temporal estricto** (`val = ttr >= '2024061600'`, últimas 2 semanas de junio de 2024, `336` horas).

| Métrica / Especificación | Valor Medido |
| :--- | :--- |
| Arquitectura | `EnsembleDownscaler` (`PhysicsUNet` + `PhysicsConvNeXt` + `PhysicsFNO`) |
| Parámetros / tamaño de los pesos | `7,561,380` parámetros · **`30.70 MB`** (`models/modelo.pt`, solo `state_dict`) |
| Score en el leaderboard público | `[Actualizar tras subir submission final]` |
| Score en validación local (`>= 2024061600`) | **`0.3821`** (reproducido en `train/submission_notebook.ipynb` vs. `0.3357` del starter) · RMSE: `RH 8.64%`, `T 0.93 K`, `U 1.62 m/s`, `V 1.54 m/s` |
| Tiempo de inferencia (24 horas de pronóstico) y hardware | **`117.4 ms`** en NVIDIA GeForce RTX 4050 (6 GB VRAM) |

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
5. **Rigor científico transparente y arranque 100% *Keyless*:** Muestra en pantalla el error real de validación de cada variable (`T ±0.93 °C`, `RH ±8.64%`, viento `±1.6 m/s`) y corre completo sin exigir cuentas ni tokens de terceros.

---

## 5. ¿Cómo usa el producto la salida de tu modelo?

El flujo opera de punta a punta sin atajos ni datos de relleno:

1. **Ingesta actual:** `app/gfs_forecast.py` descarga las últimas horas y el pronóstico GFS (24h a 48h, o fechas pasadas) desde **Open-Meteo**, guardando caché en disco (`2h`) y un respaldo offline automático para garantizar disponibilidad.
2. **Inferencia 3 km:** El backend FastAPI ejecuta **exactamente los mismos pesos (`models/modelo.pt`)** generados por `train/submission_notebook.ipynb` (vía `app/ejemplo_producto.py`), produciendo los mapas `70×35` horarios de `RH`, `T`, `U` y `V`.
3. **Cálculo de decisiones:** A partir de esas matrices de 3 km, el backend calcula máscaras de helada (`T < 0 °C`), índice de peligro de incendio, *Wind Chill*, estadísticas por polígono/ruta y trayectorias de humo advectadas por `(U, V)`, que luego se renderizan en tiempo real sobre el visor 3D (`MapLibre GL`) y en el boletín bilingüe Español/Quechua Áncash.

---

## 6. Limitaciones y próximos pasos

* **Limitaciones actuales:** (a) El error residual en vientos encajonados de valles (`RMSE ~1.54–1.62 m/s`) sigue siendo el componente más complejo por la turbulencia sub-grilla menor a 3 km; (b) el entrenamiento cubre febrero–junio (transición húmeda a seca), mientras que julio–agosto es plenamente seco, lo que introduce deriva estacional moderada; (c) en modo en vivo depende de Open-Meteo, mitigado mediante caché persistente y respaldo local.
* **Próximos pasos:** Entrenar sobre un ciclo anual completo (12 meses de simulaciones WRF), integrar estaciones meteorológicas automáticas del SENAMHI para calibración en tiempo real (*bias correction*) y conectar la exportación directa de boletines en audio a emisoras locales y WhatsApp comunitario del COER Áncash.
