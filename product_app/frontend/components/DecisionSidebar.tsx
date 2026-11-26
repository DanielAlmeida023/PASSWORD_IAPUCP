"use client";

import React, { useState } from "react";
import { AlertTriangle, CheckCircle2, Flame, ShieldAlert, Volume2 } from "lucide-react";
import type {
  BriefingResponse,
  FireHotspot,
  FirePlumeResponse,
  GeometryAnalysisResponse,
} from "../types/inti3k";

interface DecisionSidebarProps {
  analysis: GeometryAnalysisResponse | null;
  briefing: BriefingResponse | null;
  hotspots: FireHotspot[];
  firesBadge: string;
  firePlume: FirePlumeResponse | null;
  hudVisible: boolean;
  loadingAnalysis: boolean;
  onSelectHotspot: (spot: FireHotspot) => void;
  onFlyToPreset: (nombre: string, lat: number, lon: number) => void;
}

const PRESETS = [
  { nombre: "Huaraz / COER", lat: -9.528, lon: -77.528 },
  { nombre: "Caraz Agro", lat: -9.047, lon: -77.81 },
  { nombre: "Huascarán", lat: -9.122, lon: -77.604 },
  { nombre: "Pastoruri", lat: -9.927, lon: -77.193 },
  { nombre: "Chacas", lat: -9.16, lon: -77.363 },
];

export default function DecisionSidebar({
  analysis,
  briefing,
  hotspots,
  firesBadge,
  firePlume,
  hudVisible,
  loadingAnalysis,
  onSelectHotspot,
  onFlyToPreset,
}: DecisionSidebarProps) {
  const [lang, setLang] = useState<"es" | "qu">("es");

  if (!hudVisible) return null;

  const speakBriefing = () => {
    if (typeof window === "undefined" || !("speechSynthesis" in window) || !briefing) return;
    window.speechSynthesis.cancel();
    const text = lang === "es" ? briefing.spanish : briefing.quechua_ancash;
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = "es-PE";
    utt.rate = 0.96;
    window.speechSynthesis.speak(utt);
  };

  const statusColor =
    analysis?.actionable_status === "ALERTA CRÍTICA"
      ? "border-rose-500/50 bg-rose-950/50 text-rose-300"
      : analysis?.actionable_status === "PRECAUCIÓN OPERATIVA"
      ? "border-amber-500/50 bg-amber-950/50 text-amber-300"
      : "border-emerald-500/50 bg-emerald-950/50 text-emerald-300";

  return (
    <aside className="pointer-events-auto absolute top-[78px] right-3 bottom-4 z-20 flex w-[385px] max-w-[calc(100vw-24px)] flex-col gap-2.5 overflow-y-auto rounded-xl border border-sky-500/25 bg-slate-950/90 p-3.5 shadow-2xl backdrop-blur-md">
      {/* Accesos rápidos Cordillera Blanca */}
      <div>
        <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
          Vuelo Rápido Cordillera Blanca
        </div>
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((p) => (
            <button
              key={p.nombre}
              onClick={() => onFlyToPreset(p.nombre, p.lat, p.lon)}
              className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-[11px] font-medium text-slate-200 hover:border-sky-500 hover:text-sky-300"
            >
              {p.nombre}
            </button>
          ))}
        </div>
      </div>

      {/* Estado Operativo y Recomendación */}
      {loadingAnalysis ? (
        <div className="rounded-lg border border-sky-500/30 bg-slate-900/80 p-4 text-xs text-sky-300 animate-pulse">
          Consultando inferencia 3 km del modelo INTI-3K (EnsembleV8)...
        </div>
      ) : analysis ? (
        <>
          <div className={`rounded-xl border p-3 ${statusColor}`}>
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 text-sm font-extrabold">
                {analysis.actionable_status === "ALERTA CRÍTICA" ? (
                  <ShieldAlert className="h-4 w-4" />
                ) : analysis.actionable_status === "PRECAUCIÓN OPERATIVA" ? (
                  <AlertTriangle className="h-4 w-4" />
                ) : (
                  <CheckCircle2 className="h-4 w-4" />
                )}
                {analysis.actionable_status}
              </div>
              <span className="rounded bg-slate-950/60 px-2 py-0.5 text-[10px] uppercase">
                {analysis.geometry_type} · {analysis.area_km2} km²
              </span>
            </div>
            <div className="mt-1 text-xs font-semibold text-white">
              {analysis.label} <span className="font-normal text-slate-300">· Semáforo de helada</span>
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-200">{analysis.recommendation}</p>
          </div>

          {/* Alertas secundarias (incendio/sequedad, viento): cada una con su causa, celda 3 km y acción */}
          {analysis.alertas_secundarias?.length > 0 && (
            <div className="flex flex-col gap-1.5">
              {analysis.alertas_secundarias.map((a) => (
                <div
                  key={a.tipo}
                  className={`rounded-lg border p-2 text-[11px] ${
                    a.nivel === "alta"
                      ? "border-orange-500/50 bg-orange-950/40 text-orange-200"
                      : "border-yellow-600/40 bg-yellow-950/30 text-yellow-200"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 font-bold">
                    <span className="flex items-center gap-1">
                      {a.tipo === "incendio_sequedad" ? <Flame className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
                      Alerta secundaria · {a.titulo}
                    </span>
                    <span className="text-[10px] uppercase">
                      {a.nivel} · {a.valor}
                      {a.unidad}
                    </span>
                  </div>
                  <p className="mt-0.5 leading-snug text-slate-200">{a.texto}</p>
                  <p className="mt-0.5 font-semibold">{a.recomendacion}</p>
                </div>
              ))}
            </div>
          )}

          {/* Comparador explícito GFS 27 km vs INTI-3K en la celda crítica */}
          {analysis.comparador && (
            <div className="rounded-xl border border-amber-400/40 bg-gradient-to-br from-amber-950/50 to-slate-900/90 p-3">
              <div className="text-[10px] font-bold uppercase tracking-wider text-amber-300">
                Lo que el GFS 27 km no ve · tu celda 3 km
              </div>
              <div className="mt-1.5 flex items-end justify-between gap-2">
                <div>
                  <div className="text-[10px] text-slate-400">GFS 27 km decía</div>
                  <div className="text-xl font-bold text-amber-300">{analysis.comparador.gfs_27km_c}°C</div>
                </div>
                <div className="pb-1 text-slate-500">→</div>
                <div className="text-right">
                  <div className="text-[10px] text-slate-400">INTI-3K dice</div>
                  <div className="text-xl font-bold text-sky-300">{analysis.comparador.inti3k_c}°C</div>
                </div>
              </div>
              <p className="mt-1.5 text-[11px] leading-snug text-slate-200">{analysis.comparador.texto}</p>
              <p className="mt-1 text-[10px] font-semibold text-amber-200">{analysis.comparador.veredicto}</p>
              <div className="mt-1 text-[9px] text-slate-500">
                Celda fila {analysis.comparador.cell.row} · col {analysis.comparador.cell.col} de la grilla 70×35 ·{" "}
                {analysis.comparador.metodo_gfs}
              </div>
            </div>
          )}

          {/* Grid KPIs 3 km vs GFS 27 km */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-2.5">
              <div className="text-[10px] text-slate-400">Temp. Mínima 3 km ({analysis.kpis.hora_tmin_lima})</div>
              <div className="mt-0.5 text-lg font-bold text-sky-300">
                {analysis.kpis.tmin_c}°C{" "}
                <span className="text-[11px] font-normal text-slate-400">/ {analysis.kpis.tmax_c}°C</span>
              </div>
              <div className="text-[10px] text-slate-400">
                Mín. GFS 27 km en la zona: <strong className="text-amber-300">{analysis.kpis.gfs_27km_tmin_c}°C</strong> (Δ{" "}
                {analysis.kpis.delta_resolucion_c}°C)
              </div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-2.5">
              <div className="text-[10px] text-slate-400">Sensación Térmica & Viento</div>
              <div className="mt-0.5 text-lg font-bold text-slate-100">
                {analysis.kpis.wind_chill_min_c}°C{" "}
                <span className="text-[11px] font-normal text-amber-300">{analysis.kpis.viento_max_ms} m/s</span>
              </div>
              <div className="text-[10px] text-slate-400">
                Elevación DEM: {analysis.elevacion_min_m}–{analysis.elevacion_max_m} m
              </div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-2">
              <div className="text-[10px] text-slate-400">Exposición Helada (&lt;0°C, celdas 3 km)</div>
              <div className="text-sm font-bold text-cyan-300">
                {analysis.kpis.frost_hours}h · {analysis.kpis.cells_with_frost}/{analysis.cells_3km_count} celdas
              </div>
              <div className="text-[9px] text-slate-500">
                {analysis.kpis.frost_cell_hours} celda·h · celda más fría {analysis.kpis.frost_hours_worst_cell}h · GFS{" "}
                {analysis.kpis.frost_hours_gfs}h
              </div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-2">
              <div className="text-[10px] text-slate-400">Riesgo Incendio &amp; RH mín</div>
              <div className="text-sm font-bold text-orange-300">
                {analysis.kpis.fire_risk_max}/100 · RH {analysis.kpis.rh_min_pct}%
              </div>
            </div>
          </div>

          {/* Boletín Bilingüe Español / Quechua Áncash */}
          {briefing && (
            <div className="rounded-lg border border-sky-500/25 bg-slate-900/95 p-2.5">
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <div className="flex items-center gap-1 rounded bg-slate-950 p-0.5 text-[10px]">
                  <button
                    onClick={() => setLang("es")}
                    className={`rounded px-2 py-0.5 font-semibold ${
                      lang === "es" ? "bg-sky-600 text-white" : "text-slate-400"
                    }`}
                  >
                    Español
                  </button>
                  <button
                    onClick={() => setLang("qu")}
                    className={`rounded px-2 py-0.5 font-semibold ${
                      lang === "qu" ? "bg-sky-600 text-white" : "text-slate-400"
                    }`}
                  >
                    Quechua Áncash
                  </button>
                </div>
                <button
                  onClick={speakBriefing}
                  className="flex items-center gap-1 rounded bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-sky-300 hover:bg-slate-700"
                  title="Leer boletín en voz alta"
                >
                  <Volume2 className="h-3 w-3" /> Escuchar
                </button>
              </div>
              <p className="text-[11px] leading-relaxed text-slate-200">
                {lang === "es" ? briefing.spanish : briefing.quechua_ancash}
              </p>
            </div>
          )}

          {/* Cono de Humo / Incendios Activos */}
          <div className="rounded-lg border border-orange-500/25 bg-slate-900/90 p-2.5">
            <div className="flex items-center justify-between text-[10px] font-bold uppercase text-orange-300 mb-1.5">
              <span className="flex items-center gap-1">
                <Flame className="h-3.5 w-3.5" /> Focos &amp; Cono de Humo 3 km
              </span>
              <span className="text-[9px] font-normal text-slate-400">
                {hotspots[0]?.source === "NASA_FIRMS_LIVE" ? "NASA FIRMS LIVE" : "Simulador Keyless"}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 mb-1.5">{firesBadge}</div>
            <div className="flex flex-col gap-1">
              {hotspots.slice(0, 3).map((spot) => (
                <button
                  key={spot.id}
                  onClick={() => onSelectHotspot(spot)}
                  className="flex items-center justify-between rounded bg-slate-950/80 px-2 py-1 text-left text-[11px] text-slate-200 hover:bg-orange-950/50"
                >
                  <span className="truncate pr-2">{spot.nombre}</span>
                  <span className="shrink-0 text-[10px] text-orange-300">Proyectar cono →</span>
                </button>
              ))}
            </div>
            {firePlume && (
              <div className="mt-2 rounded bg-orange-950/40 p-2 text-[10px] text-orange-200">
                Cono activo ({firePlume.duration_hours}h desde {firePlume.start_time_lima}): Riesgo promedio{" "}
                <strong>{firePlume.avg_fire_risk}/100</strong>, ráfaga máx{" "}
                <strong>{firePlume.max_wind_ms} m/s</strong>.
              </div>
            )}
          </div>

          {/* Perfil de Ruta si geometry_type === "route" */}
          {analysis.route_profile && analysis.route_profile.length > 0 && (
            <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-2.5">
              <div className="text-[10px] font-bold uppercase text-sky-300 mb-1.5">
                Perfil de Ruta ({analysis.route_distance_km} km)
              </div>
              <div className="max-h-32 overflow-y-auto text-[10px]">
                <table className="w-full text-left">
                  <thead className="text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="py-1">Km</th>
                      <th>Elev</th>
                      <th>T mín</th>
                      <th>Chill</th>
                      <th>Viento</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.route_profile.map((pt, idx) => (
                      <tr key={idx} className="border-b border-slate-800/60">
                        <td className="py-1">{pt.dist_km}</td>
                        <td>{pt.elevacion_m}m</td>
                        <td className={pt.tmin_c <= 0 ? "text-sky-300 font-bold" : ""}>{pt.tmin_c}°C</td>
                        <td>{pt.wind_chill_min_c}°C</td>
                        <td>{pt.wmax_ms} m/s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Serie horaria compacta */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-2.5">
            <div className="text-[10px] font-bold uppercase text-slate-400 mb-1.5">
              Evolución Horaria (hora Perú, UTC−5) · celda crítica ({analysis.comparador?.cell.lat},{" "}
              {analysis.comparador?.cell.lon})
            </div>
            <div className="max-h-40 overflow-y-auto text-[10px]">
              <table className="w-full text-left">
                <thead className="text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-1">Hora</th>
                    <th>INTI-3K celda</th>
                    <th>GFS 27km celda</th>
                    <th>RH</th>
                    <th>Viento</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.hourly_series.slice(0, 48).map((h) => (
                    <tr key={h.time_utc} className="border-b border-slate-800/60">
                      <td className="py-1 font-medium text-slate-300">{h.time_lima}</td>
                      <td className={h.temp_cell_c < 0 ? "font-bold text-sky-300" : "text-slate-100"}>
                        {h.temp_cell_c}°C
                      </td>
                      <td className="text-slate-400">{h.temp_gfs_cell_c}°C</td>
                      <td>{h.rh_min_pct}%</td>
                      <td>{h.wind_max_ms} m/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : null}
    </aside>
  );
}
