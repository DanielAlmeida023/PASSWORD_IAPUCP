# Reporte del producto · PASSWORD

<!-- TODO EQUIPO: nombres completos de los integrantes y link público del video -->
**Integrantes:** >>> TODO EQUIPO: Nombre 1, Nombre 2, Nombre 3, Nombre 4 <<<  
**Producto:** INTI-3K · se ejecuta en local, ver [README](./README.md) (3 pasos, sin claves de API)  
**Video pitch:** >>> TODO EQUIPO: LINK DEL VIDEO <<<

---

## 1. ¿Qué modelo o arquitectura usaron?

<!-- ACTUALIZAR CON MODELO FINAL: arquitectura, tabla y números de esta sección -->

**`EnsembleV8`**: un solo archivo de pesos (`models/modelo.pt`) con cuatro redes. Cada una predice una corrección sobre una base física, y se mezclan con pesos fijos por variable:

- **ConvNeXt** con kernels 7×7 dilatados (detalle fino del relieve),
- **multi-escala** (bloques 3×3 + 7×7 en paralelo),
- **U-Net** de 3 niveles (circulación de valle; la mejor en viento),
- **FNO 2D** compacto (patrones amplios de humedad).

Las cuatro parten de un **ancla física**: el GFS interpolado a 3 km más un gradiente de temperatura y de humedad con la altura, **aprendido por hora UTC**. La red solo aprende lo que esa base no explica. Un bloque de atención temporal mezcla los lags y otro de atención cruzada deja que cada celda de 3 km consulte las celdas GFS vecinas.

- **Entradas** (todas derivadas de la data oficial): GFS 17×13 en los lags −24, −6, −3, −1, 0, +1, +3, +6 y +24 h; media y desviación de 24 h; gradientes de T y RH, divergencia y vorticidad del viento; punto de rocío; hora UTC; 10 canales de relieve del DEM (altura, pendientes, forzamiento orográfico viento·∇DEM).
- **Pérdida:** RMSE de cada variable dividido por el RMSE de la climatología horaria, o sea, la métrica oficial en forma diferenciable. Las semanas recientes pesan más, y se promedian los pesos con EMA.
- **Validación:** corte temporal, nunca aleatorio. Fase A: train hasta el 15 jun, val del 16 al 30 jun (336 h), para elegir épocas y pesos de mezcla. Fase B: reentrenamiento con el 100 % de febrero–junio (2,935 h) con esas épocas fijas. También hay un CV de 3 folds (16–31 may, 1–15 jun, 16–30 jun, con 24 h de separación) en `kaggle_pipeline/src/cv.py`; para EnsembleV8 solo está registrado el fold 3. <!-- ACTUALIZAR CON MODELO FINAL -->
- **Ojo con el 0.4100:** es optimista, porque la época de cada red se eligió en esa misma validación. El LB público mide otro período (julio–agosto).

| | |
|---|---|
| Arquitectura | `EnsembleV8`: ConvNeXt + multi-escala + U-Net + FNO, residuales sobre un ancla física, mezcla por variable |
| Parámetros / tamaño de los pesos | 5,741,468 valores en el `state_dict` · **23.11 MB** (`models/modelo.pt`, sin estado del optimizador) |
| Score en el leaderboard público | **0.49844** <!-- ACTUALIZAR CON MODELO FINAL --> |
| Score en tu validación local | **0.4100** <!-- ACTUALIZAR CON MODELO FINAL --> (holdout 16–30 jun; starter: 0.3357 en el mismo corte). RH 0.510 · T 0.587 · U 0.308 · V 0.235. RMSE: T 0.83 K, RH 8.4 %, U 1.56 m/s, V 1.49 m/s |
| Tiempo de inferencia (24 horas de pronóstico) y hardware | 0.19 s en GPU NVIDIA RTX 4050 Laptop (6 GB) · ~4 s en CPU de laptop (incluye el cálculo de features) |

---

## 2. ¿Para quién es y qué problema le resuelve?

**Usuario:** el productor de papa (y la cooperativa o el técnico agrario que lo asesora) del Callejón de Huaylas y de las partes altas de las cordilleras Blanca y Negra, entre 3,000 y 4,200 m. **Decisión:** en la tarde, *¿protejo esta noche mi chacra contra la helada (riego, cobertura, humo) o no hace falta?*

Hoy decide con el pronóstico del celular, que sale del GFS de ~27 km: un solo número para una celda que mezcla el valle a 2,500 m con la puna a 4,500 m. Eso falla en las dos direcciones. Ejemplo con el GFS real de la noche del **18 jul 2024** (salida de nuestro producto):

- **Parcela a 4,000 m en la Cordillera Negra, frente a Huaraz** (−9.563, −77.650): *“GFS 27 km decía 4.6 °C, INTI-3K dice −0.8 °C en tu celda 3 km el jue 18/07 05:00 (hora Perú)”*. Son 3 h bajo 0 °C que el GFS no veía; sin aviso, la chacra no se protege.
- **Valle en Marcará, 3,081 m** (−9.320, −77.568): el GFS interpolado marcaba 10 h bajo 0 °C; INTI-3K da una mínima de 5.2 °C y 0 h de helada. El productor se ahorra regar de madrugada, con su gasto de agua y mano de obra.

Con INTI-3K decide con **la mínima, la hora y las horas bajo 0 °C de su propia celda de 3 km**, antes del anochecer y en hora Perú.

Usuarios secundarios con el mismo motor: COER Áncash y municipios (polígono por distrito), guías de montaña (ruta con sensación térmica), brigadas forestales (cono de humo).

---

## 3. ¿Cuál es la propuesta de valor del producto?

**Saber esta tarde si tu chacra tendrá helada esta noche, a 3 km y no a 27 km, para proteger solo cuando hace falta.** En los dos casos de julio 2024 que trae la demo, el GFS marcaba helada en el 15–27 % de las celdas entre 2,500 y 3,500 m, y nuestro modelo en el 0.5 %. Por encima de 4,500 m pasa lo contrario: INTI-3K ve helada en ~80 % de las celdas y el GFS en ~60–65 %. Son menos falsas alarmas en el valle y menos heladas no vistas en la puna.

Esos porcentajes comparan dos pronósticos entre sí: la data oficial no trae verdad de terreno para julio. La evidencia de que el modelo acierta es la métrica: skill 0.41 en validación y 0.498 en el LB público, ambos por encima de la climatología.

---

## 4. ¿Por qué tu producto es diferente a algo usual?

Una app de clima da un número por ciudad, y un mapa del pronóstico da colores. INTI-3K responde **una pregunta sobre una parcela concreta** y muestra cuánto se habría equivocado el pronóstico global:

1. **Comparador en tu celda:** cada alerta cita la celda de 3 km (lat, lon, altura) con el valor del modelo junto al GFS interpolado, y dice si el GFS *no veía* la helada o la *marcaba de más*. Con la tecla `G` se ve el mismo contraste en todo el mapa 3D.
2. **Tu parcela, no tu ciudad:** el usuario marca un pin o dibuja su terreno, y el producto calcula horas bajo 0 °C, celdas afectadas y área en km² sobre la grilla 70×35.
3. **Casos reales para generar confianza:** con un clic se reproduce la helada del 18 o del 29 de julio de 2024 con el GFS histórico, para que el productor compare con lo que vivió.
4. **Pensado para la sierra:** no pide claves ni cuentas. Si cae la conexión usa la última descarga y lo dice en pantalla. El boletín sale en español y en quechua de Áncash, con lectura en voz alta.

---

## 5. ¿Cómo usa el producto la salida de tu modelo?

1. **GFS actual:** `product_app/backend/app/gfs_cache_manager.py` llama a `app/gfs_forecast.py` (Open-Meteo, modelo `gfs_global`) con 24 h pasadas y 78 h futuras, porque el modelo usa lags de −24 a +24 h. La descarga se guarda en `gfs_fallback_cache.npz` y se reutiliza 3 h (configurable de 1 a 6 h). Si Open-Meteo falla, se usa la última caché sin importar su edad y la UI lo avisa. Para fechas pasadas usa la API histórica del mismo GFS y guarda el resultado en disco.
2. **Modelo:** `app/inti_inference.py` arma las features con **las mismas funciones del entrenamiento** (`train_v7.py`, `train.py`), normaliza con `models/norm_stats.npz` y corre `EnsembleV8` con `models/modelo.pt`, el mismo archivo que generó la submission. Un test (`product_app/backend/tests/test_lags.py`) verifica que las ventanas de lags sean idénticas a las del entrenamiento. Salida: `RH`, `T`, `U`, `V` por hora en 70×35 celdas.
3. **Decisión:** el backend (`main.py`) pasa la hora UTC a hora Perú (UTC−5). Sobre las celdas de 3 km del pin o del polígono calcula la mínima y su hora, las horas bajo 0 °C, el comparador contra el GFS interpolado (sin modelo) y un semáforo (ALERTA / PRECAUCIÓN / VENTANA SEGURA). La recomendación siempre cita la celda y el valor predicho. Sin el modelo solo quedaría el GFS borroso, sin comparador ni alerta por celda.
4. **Herramientas alrededor:**
   - **Boletín español / quechua:** se arma con **plantillas fijas** en las que se insertan los valores del modelo. El equipo redactó el texto en quechua de Áncash; **no lo ha revisado un hablante nativo y no es una traducción certificada**. <!-- TODO EQUIPO: si un hablante lo revisa, indicarlo aquí -->
   - **LLM opcional:** si hay un **Ollama** local (`OLLAMA_URL`), se usa solo para resumir el texto en español en 2 oraciones. El quechua no pasa por el LLM. Sin Ollama se usa la plantilla, y la respuesta lo indica en el campo `generator`.
   - **Voz:** la síntesis de voz del navegador lee el boletín.
   - **NASA FIRMS (opcional, clave gratuita):** focos de incendio reales para el cono de humo, que avanza con el viento `U, V` del modelo. Sin clave se usa un simulador.
   - **Mapa 3D:** Esri World Imagery y relieve AWS Terrarium, sin claves.

---

## 6. Limitaciones y próximos pasos

**Dónde es menos confiable:**
- **Viento:** es la variable más débil (skill U 0.31, V 0.24; error ~1.5 m/s). El cono de humo es solo orientativo.
- **Periodo de entrenamiento:** el WRF de entrenamiento cubre febrero–junio de 2024, así que la temporada de heladas (julio–agosto) no tiene verdad de terreno en la data. El LB público sugiere que el modelo generaliza, pero no lo garantiza para otros años.
- **Límites físicos:** el modelo no corrige errores grandes del propio GFS (si el GFS falla la masa de aire, el modelo también). Lo que pasa bajo 3 km (heladas de hondonada, una sola parcela) sigue sin verse.
- **Semáforo:** combina helada, viento e índice de incendio; el índice es una fórmula heurística, no sale del modelo. Puede salir “ALERTA” por incendio en una noche sin helada, y el texto de la recomendación todavía no separa bien los dos casos. <!-- TODO EQUIPO: borrar este punto si se corrige en el código -->
- **Sin validar en campo:** no se ha contrastado con estaciones del SENAMHI, y el quechua no lo han validado hablantes.

**Próximos pasos:**
- Validar con estaciones del SENAMHI y con productores de una cooperativa en la próxima temporada de heladas.
- Alertas por WhatsApp/SMS a las 5 p. m. con la mínima de cada parcela registrada.
- Revisión del quechua con hablantes de Huaraz y Carhuaz.
- Umbrales por cultivo y fase (papa en floración vs. tuberización).
- Entrenar con más meses de WRF si el proyecto AIRE-ANDINO los libera.
