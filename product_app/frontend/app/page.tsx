"use client";

import React, { useState, useEffect } from "react";

interface HourlyPoint {
  time_utc: string;
  time_lima: string;
  temp_c: number;
  rh_pct: number;
  wind_ms: number;
  u_ms: number;
  v_ms: number;
}

interface DownscaleResponse {
  timestamp_utc: string;
  periodo_lima: string;
  model_info: { exp_id: string; val_skill_score: number };
  zona: {
    id: string;
    nombre: string;
    perfil: string;
    lat_real_3km: number;
    lon_real_3km: number;
    elevacion_m: number;
  };
  kpis_zona: {
    tmin_c: number;
    tmax_c: number;
    hora_tmin_lima: string;
    viento_max_ms: number;
    rh_media_pct: number;
  };
  resumen_cordillera: {
    total_celdas_3km: number;
    pct_territorio_helada: number;
    tmin_dominio_c: number;
    tmax_dominio_c: number;
    viento_max_dominio_ms: number;
  };
  actionable_status: string;
  recommendation: string;
  serie_horaria_3km: HourlyPoint[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const ZONAS = [
  { id: "huaraz_valle", label: "Huaraz · Valle del Santa (Agricultura & Ciudad)" },
  { id: "caraz_cultivos", label: "Caraz · Campos Agrícolas (Agroexportación)" },
  { id: "huascaran_ruta", label: "Nevado Huascarán · Campo Base (Andinismo)" },
  { id: "pastoruri_turismo", label: "Glaciar Pastoruri · Ruta Turística (Turismo)" },
  { id: "chacas_vertiente", label: "Chacas · Vertiente Oriental (Gestión Municipal)" },
];

export default function CordilleraDashboard() {
  const [zonaId, setZonaId] = useState("huaraz_valle");
  const [forecastHours, setForecastHours] = useState(24);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<DownscaleResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchDownscale = async (selectedZona = zonaId, hours = forecastHours) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/downscale-live`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          zona_id: selectedZona,
          forecast_hours: hours,
          umbral_helada_c: 0.0,
          umbral_viento_ms: 10.0,
        }),
      });
      if (!res.ok) {
        throw new Error(`Error ${res.status}: Verifica que el backend FastAPI esté corriendo en el puerto 8000.`);
      }
      const json: DownscaleResponse = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message || "Error al conectar con la API de Downscaling.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDownscale(zonaId, forecastHours);
  }, []);

  return (
    <main style={{ minHeight: "100vh", backgroundColor: "#070b14", color: "#f1f5f9", padding: "2rem", fontFamily: "system-ui, sans-serif" }}>
      <div style={{ maxWidth: "1180px", margin: "0 auto" }}>
        <header style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "1rem", borderBottom: "1px solid #1e293b", paddingBottom: "1.25rem", marginBottom: "1.75rem" }}>
          <div>
            <span style={{ fontSize: "0.75rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "#38bdf8", fontWeight: 700 }}>
              Hackathon IA PUCP 2026 · Proyecto AIRE-ANDINO (GFS 27 km → Modelo IA 3 km)
            </span>
            <h1 style={{ fontSize: "1.85rem", fontWeight: 800, margin: "0.25rem 0 0 0" }}>
              Inteligencia Microclimática 3 km — Cordillera Blanca
            </h1>
          </div>

          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
            <select
              value={zonaId}
              onChange={(e) => {
                setZonaId(e.target.value);
                fetchDownscale(e.target.value, forecastHours);
              }}
              style={{ backgroundColor: "#111827", border: "1px solid #334155", color: "#f8fafc", padding: "0.55rem 0.9rem", borderRadius: "0.5rem" }}
            >
              {ZONAS.map((z) => (
                <option key={z.id} value={z.id}>
                  {z.label}
                </option>
              ))}
            </select>
            <button
              onClick={() => fetchDownscale(zonaId, forecastHours)}
              disabled={loading}
              style={{ backgroundColor: "#0284c7", color: "#fff", fontWeight: 600, padding: "0.55rem 1.2rem", borderRadius: "0.5rem", border: "none", cursor: "pointer" }}
            >
              {loading ? "Ejecutando Modelo 3 km..." : "Actualizar Pronóstico GFS Vivo"}
            </button>
          </div>
        </header>

        {error && (
          <div style={{ backgroundColor: "#450a0a", border: "1px solid #dc2626", padding: "1rem", borderRadius: "0.5rem", marginBottom: "1.5rem" }}>
            {error}
          </div>
        )}

        {data && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1.25rem", marginBottom: "1.5rem" }}>
              <div style={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", borderRadius: "0.75rem", padding: "1.25rem" }}>
                <div style={{ fontSize: "0.78rem", color: "#94a3b8", textTransform: "uppercase" }}>Decisión Operativa ({data.zona.perfil})</div>
                <div
                  style={{
                    fontSize: "1.65rem",
                    fontWeight: 800,
                    margin: "0.5rem 0",
                    color: data.actionable_status === "ALERTA CRÍTICA" ? "#f87171" : data.actionable_status === "PRECAUCIÓN OPERATIVA" ? "#fbbf24" : "#34d399",
                  }}
                >
                  {data.actionable_status}
                </div>
                <p style={{ fontSize: "0.9rem", color: "#cbd5e1", lineHeight: 1.45, margin: 0 }}>{data.recommendation}</p>
              </div>

              <div style={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", borderRadius: "0.75rem", padding: "1.25rem" }}>
                <div style={{ fontSize: "0.78rem", color: "#94a3b8", textTransform: "uppercase" }}>
                  Celda 3 km · {data.zona.nombre}
                </div>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0.4rem 0" }}>
                  {data.kpis_zona.tmin_c} °C <span style={{ fontSize: "0.95rem", color: "#64748b" }}>/ máx {data.kpis_zona.tmax_c} °C</span>
                </div>
                <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
                  Elevación DEM: <strong>{data.zona.elevacion_m} m s.n.m.</strong> ({data.zona.lat_real_3km}°, {data.zona.lon_real_3km}°)
                </div>
                <div style={{ fontSize: "0.85rem", color: "#38bdf8", marginTop: "0.35rem" }}>
                  Mínima estimada: {data.kpis_zona.hora_tmin_lima} (Hora Perú)
                </div>
              </div>

              <div style={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", borderRadius: "0.75rem", padding: "1.25rem" }}>
                <div style={{ fontSize: "0.78rem", color: "#94a3b8", textTransform: "uppercase" }}>Viento & Humedad a 3 km</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0.4rem 0" }}>
                  {data.kpis_zona.viento_max_ms} m/s <span style={{ fontSize: "0.95rem", color: "#64748b" }}>· RH {data.kpis_zona.rh_media_pct}%</span>
                </div>
                <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
                  Cobertura de heladas en Cordillera (70×35): <strong>{data.resumen_cordillera.pct_territorio_helada}%</strong>
                </div>
                <div style={{ fontSize: "0.8rem", color: "#64748b", marginTop: "0.35rem" }}>
                  Modelo Kaggle: <code>{data.model_info.exp_id}</code> (Skill Score Val: {data.model_info.val_skill_score})
                </div>
              </div>
            </div>

            <section style={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", borderRadius: "0.75rem", padding: "1.5rem" }}>
              <h2 style={{ fontSize: "1rem", fontWeight: 700, marginTop: 0, marginBottom: "1rem", color: "#e2e8f0" }}>
                Evolución Horaria a 3 km ({data.periodo_lima} · Hora de Lima UTC-5)
              </h2>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #334155", textAlign: "left", color: "#94a3b8" }}>
                      <th style={{ padding: "0.6rem" }}>Hora Lima</th>
                      <th style={{ padding: "0.6rem" }}>Temp (°C)</th>
                      <th style={{ padding: "0.6rem" }}>Humedad RH (%)</th>
                      <th style={{ padding: "0.6rem" }}>Viento Total (m/s)</th>
                      <th style={{ padding: "0.6rem" }}>U Oeste→Este</th>
                      <th style={{ padding: "0.6rem" }}>V Sur→Norte</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.serie_horaria_3km.map((h) => (
                      <tr key={h.time_utc} style={{ borderBottom: "1px solid #1e293b" }}>
                        <td style={{ padding: "0.55rem 0.6rem", fontWeight: 600 }}>{h.time_lima}</td>
                        <td style={{ padding: "0.55rem 0.6rem", color: h.temp_c <= 0 ? "#60a5fa" : "#f8fafc", fontWeight: h.temp_c <= 0 ? 700 : 400 }}>
                          {h.temp_c} °C
                        </td>
                        <td style={{ padding: "0.55rem 0.6rem" }}>{h.rh_pct}%</td>
                        <td style={{ padding: "0.55rem 0.6rem", fontWeight: 600 }}>{h.wind_ms} m/s</td>
                        <td style={{ padding: "0.55rem 0.6rem", color: "#94a3b8" }}>{h.u_ms}</td>
                        <td style={{ padding: "0.55rem 0.6rem", color: "#94a3b8" }}>{h.v_ms}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
