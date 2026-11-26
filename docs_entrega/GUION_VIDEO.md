# Guion del video pitch · INTI-3K (≤ 5:00)

**Regla:** poca diapositiva, mucha demo. Un solo usuario (productor de papa) y una sola pregunta: *¿protejo mi chacra esta noche?* Todo lo demás es secundario y, si falta tiempo, se corta.

**Antes de grabar (checklist):**
- [ ] Backend y frontend corriendo (`start_local.ps1` + `npm run dev`), navegador a pantalla completa en `http://localhost:3000`.
- [ ] Abrir la app una vez antes de grabar, para que el GFS actual quede descargado y la primera inferencia ya esté hecha (no grabar la espera).
- [ ] Probar los dos pins de abajo con el caso **Helada 18 jul 2024** y anotar los números que salgan.
- [ ] Hora del sistema visible o mencionada: toda la UI está en **hora Perú (UTC−5)**.
- [ ] Plan B: si Open-Meteo falla en vivo, no cortar. El HUD muestra “SIN CONEXIÓN · usando caché GFS”; decirlo en voz alta, porque también es una función.
- [ ] Subir el video como público o no listado, con el link abierto a cualquiera, y pegarlo en `README.md` y `REPORTE_PRODUCTO.md`.

---

## 0:00 – 0:30 · Problema (30 s)

| Pantalla | Voz (aprox.) |
|---|---|
| Mapa 3D de la Cordillera Blanca en vista satélite (`1`), girando lento. Texto sobreimpreso: *“Callejón de Huaylas · papa entre 3,000 y 4,200 m”*. | “En el Callejón de Huaylas, una helada en una noche puede quemar la papa de toda una campaña. Para decidir si riega o cubre, el productor mira el pronóstico del celular.” |
| Pulsar `G` → mapa en **GFS 27 km** (manchas grandes). | “Ese pronóstico sale del GFS: celdas de 27 km. Un solo número para el valle a 2,500 m y la puna a 4,500 m. Se equivoca hacia los dos lados: da falsas alarmas en el valle y no ve heladas arriba.” |

## 0:30 – 3:30 · Demo en vivo (3 min)

### 0:30 – 1:30 · Hoy, con el GFS actual (60 s)

| Pantalla | Voz (aprox.) |
|---|---|
| HUD superior: badge verde **“GFS actual en vivo (Open-Meteo) · bajado … (hora Perú)”** y badge del modelo con su skill. Hacer zoom al badge. | “Esto es INTI-3K corriendo ahora, con el GFS de hoy bajado de Open-Meteo. Nuestro modelo de Kaggle lo baja a 3 km en una fracción de segundo.” |
| Pulsar `G` varias veces: GFS 27 km ↔ INTI-3K 3 km en vista térmica (`2`). | “Así se ve la diferencia: a la izquierda lo que ve el GFS, a la derecha nuestro mapa de 3 km, valle por valle.” |
| Botón **Pin** → clic sobre una chacra en altura (ej. Cordillera Negra frente a Huaraz, ~4,000 m). Panel derecho: mínima, hora, horas bajo 0 °C, semáforo. | “Soy productor, marco mi chacra. Esta noche tengo una mínima de `[X] °C` a las `[hora]`, hora Perú.” <!-- TODO EQUIPO: completar con lo que salga al grabar --> |
| Zoom al recuadro **“Lo que el GFS 27 km no ve · tu celda 3 km”**. | “Y aquí está el punto: el GFS decía `[A] °C`, INTI-3K dice `[B] °C` en mi celda de 3 km.” <!-- TODO EQUIPO --> |
| `Espacio` para reproducir las horas en la línea de tiempo. | (Sin voz o “hora por hora, toda la noche”.) |

### 1:30 – 2:45 · Caso real: helada de julio 2024 (75 s)

| Pantalla | Voz (aprox.) |
|---|---|
| Línea de tiempo → selector de casos → **“Helada 18 jul 2024”**. Badge violeta **“Fecha pasada · GFS histórico”**. | “Ahora, una noche real: el 18 de julio de 2024, una de las noches más frías de julio, con el GFS que existía ese día.” |
| Pin en **(−9.563, −77.650), ~4,000 m, Cordillera Negra**. Zoom al comparador. | “En esta chacra a 4,000 metros el GFS decía **4.6 °C**: dormir tranquilo. INTI-3K dice **−0.8 °C a las 5 a. m.**, con 3 horas bajo cero. El veredicto lo dice claro: *el GFS no veía helada en esta celda; INTI-3K sí la detecta*.” |
| Pin en **Marcará (−9.320, −77.568), 3,081 m**. | “Y abajo, en Marcará, el GFS marcaba 10 horas de helada. INTI-3K da una mínima de **5.2 °C**. Ese productor no necesita levantarse a regar a las 3 a. m.” |
| (Opcional, 5 s) Vista `4` riesgo con polígono de un distrito. | “Lo mismo sirve para un distrito entero, para el COER o el municipio.” |

*Números verificados con el modelo actual el 26 nov 2026.* <!-- ACTUALIZAR CON MODELO FINAL: volver a correr los dos pins y corregir las cifras -->

### 2:45 – 3:30 · La alerta que le llega al productor (45 s)

| Pantalla | Voz (aprox.) |
|---|---|
| Panel derecho → boletín en **Español**. Resaltar la cita de la celda en el texto. | “El producto arma la alerta con la acción concreta, y siempre cita la celda y el valor que predijo el modelo.” |
| Cambiar a **Quechua Áncash** → **Escuchar** (dejar sonar 5–8 s). | “También en quechua de Áncash, leída en voz alta para la radio o el WhatsApp de la comunidad. Es una plantilla escrita por el equipo; queremos validarla con hablantes.” |
| (Si da tiempo) desconectar el wifi y recargar: badge **“SIN CONEXIÓN · usando caché GFS”**. | “Y si en la sierra se cae internet, sigue funcionando con la última descarga, y lo avisa.” |

## 3:30 – 4:30 · Modelo y score (1 min)

| Pantalla | Voz (aprox.) |
|---|---|
| Diapositiva única o README: diagrama *GFS 17×13 (lags −24…+24 h) + relieve → EnsembleV8 (ConvNeXt · multi-escala · U-Net · FNO sobre un ancla física) → 70×35 a 3 km*. | “El modelo es EnsembleV8: cuatro redes que parten de una base física, el GFS más el enfriamiento con la altura, y aprenden la corrección valle por valle a partir del WRF del proyecto AIRE-ANDINO.” |
| Cifras grandes: **LB público 0.49844** · **validación 0.4100** (16–30 jun) · **23 MB** · **0.19 s / 24 h en GPU**. <!-- ACTUALIZAR CON MODELO FINAL --> | “En el leaderboard público sacamos 0.498 de skill; en nuestra validación temporal, 0.41. Esa cifra es optimista porque elegimos las épocas en ese mismo corte. Los pesos pesan 23 MB y un día de pronóstico sale en menos de un segundo.” |
| Captura de `train/submission_notebook.ipynb` (celda `TRAIN = True`) y de `pytest` en verde. | “Todo es reproducible: un notebook entrena y genera el submission, y el producto carga exactamente esos pesos. Tenemos tests que verifican que el producto prepara los datos igual que el entrenamiento.” |

## 4:30 – 5:00 · Impacto (30 s)

| Pantalla | Voz (aprox.) |
|---|---|
| Vuelta al mapa 3D con la helada del 18 jul en vista térmica; texto: *“Proteger solo cuando hace falta”*. | “INTI-3K le cambia la pregunta al productor: ya no es ‘¿hará frío en Huaraz?’, sino ‘¿helará esta noche en mi chacra?’. Protege cuando hace falta y no gasta cuando no.” |
| Cierre: nombre del equipo, link del repo, integrantes. <!-- TODO EQUIPO: nombres y URL --> | “Somos el equipo PASSWORD. INTI-3K: el pronóstico de 27 km, a la medida de tu chacra.” |

---

**Si hay que recortar** (en este orden): el polígono de distrito → la prueba sin wifi → la reproducción con `Espacio`. **No recortar nunca:** el comparador “GFS decía / INTI-3K dice” y el caso del 18 jul.
