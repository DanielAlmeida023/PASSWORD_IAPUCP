"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import CordilleraMap3D from "../components/CordilleraMap3D";
import TacticalHUD from "../components/TacticalHUD";
import TimelineScrubber from "../components/TimelineScrubber";
import DecisionSidebar from "../components/DecisionSidebar";
import type {
  BriefingResponse,
  DrawMode,
  FireHotspot,
  FirePlumeResponse,
  FiresFirmsResponse,
  GeometryAnalysisResponse,
  GridForecastResponse,
  OpticMode,
} from "../types/inti3k";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Inti3kCommandCenter() {
  const [gridData, setGridData] = useState<GridForecastResponse | null>(null);
  const [infrastructure, setInfrastructure] = useState<GeoJSON.FeatureCollection | null>(null);
  const [hotspots, setHotspots] = useState<FireHotspot[]>([]);
  const [firesBadge, setFiresBadge] = useState<string>("");
  const [firePlume, setFirePlume] = useState<FirePlumeResponse | null>(null);

  const [hourIdx, setHourIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1);
  const [forecastHours, setForecastHours] = useState(24);
  const [historicalDate, setHistoricalDate] = useState("");

  const [opticMode, setOpticMode] = useState<OpticMode>("flir");
  const [compareGfs, setCompareGfs] = useState(false);
  const [drawMode, setDrawMode] = useState<DrawMode>("pin");
  const [draftCoords, setDraftCoords] = useState<number[][]>([[-9.528, -77.528]]);
  const [hudVisible, setHudVisible] = useState(true);

  const [analysis, setAnalysis] = useState<GeometryAnalysisResponse | null>(null);
  const [briefing, setBriefing] = useState<BriefingResponse | null>(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const [flyToTarget, setFlyToTarget] = useState<{ lat: number; lon: number; zoom?: number; seq: number } | null>(
    null
  );
  const [resetSeq, setResetSeq] = useState(0);

  const [voiceListening, setVoiceListening] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);

  const runGeometryAnalysis = useCallback(
    async (
      geomType: "pin" | "polygon" | "route",
      coords: number[][],
      label: string,
      hoursOverride?: number,
      histDateOverride?: string
    ) => {
      setLoadingAnalysis(true);
      try {
        const hrs = hoursOverride ?? forecastHours;
        const hDate = histDateOverride !== undefined ? histDateOverride : historicalDate;
        const res = await fetch(`${API_BASE}/api/analyze-geometry`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            geometry_type: geomType,
            coordinates: coords,
            label,
            forecast_hours: hrs,
            start_date: hDate || null,
            end_date: hDate || null,
          }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: GeometryAnalysisResponse = await res.json();
        setAnalysis(data);

        // Generar boletín bilingüe Español / Quechua Áncash
        const bRes = await fetch(`${API_BASE}/api/briefing`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            label: data.label,
            geometry_type: data.geometry_type,
            elevacion_min_m: data.elevacion_min_m,
            elevacion_max_m: data.elevacion_max_m,
            tmin_c: data.kpis.tmin_c,
            tmax_c: data.kpis.tmax_c,
            hora_tmin_lima: data.kpis.hora_tmin_lima,
            viento_max_ms: data.kpis.viento_max_ms,
            wind_chill_min_c: data.kpis.wind_chill_min_c,
            rh_min_pct: data.kpis.rh_min_pct,
            fire_risk_max: data.kpis.fire_risk_max,
            actionable_status: data.actionable_status,
            area_km2: data.area_km2,
          }),
        });
        if (bRes.ok) {
          setBriefing(await bRes.json());
        }
      } catch (err: any) {
        setApiError(err.message || "Error analizando geometría");
      } finally {
        setLoadingAnalysis(false);
      }
    },
    [forecastHours, historicalDate]
  );

  const loadGridAndContext = useCallback(
    async (hrs: number, histDate: string) => {
      setApiError(null);
      try {
        const params = new URLSearchParams({ hours: String(hrs) });
        if (histDate) {
          params.set("start_date", histDate);
          params.set("end_date", histDate);
        }
        const [gridRes, infraRes, firesRes] = await Promise.all([
          fetch(`${API_BASE}/api/grid-forecast?${params.toString()}`),
          fetch(`${API_BASE}/api/infrastructure`),
          fetch(`${API_BASE}/api/fires-firms`),
        ]);
        if (!gridRes.ok) throw new Error("No se pudo cargar /api/grid-forecast. Verifica el backend en :8000.");
        const gridJson: GridForecastResponse = await gridRes.json();
        setGridData(gridJson);
        setHourIdx(0);

        if (infraRes.ok) setInfrastructure(await infraRes.json());
        if (firesRes.ok) {
          const fj: FiresFirmsResponse = await firesRes.json();
          setHotspots(fj.hotspots);
          setFiresBadge(fj.badge);
        }
      } catch (err: any) {
        setApiError(err.message || "Error conectando con el backend INTI-3K.");
      }
    },
    []
  );

  useEffect(() => {
    loadGridAndContext(forecastHours, historicalDate);
    runGeometryAnalysis("pin", [[-9.528, -77.528]], "Huaraz · COER Áncash", forecastHours, historicalDate);
  }, [forecastHours, historicalDate, loadGridAndContext, runGeometryAnalysis]);

  // Reproductor temporal automático
  useEffect(() => {
    if (!isPlaying || !gridData?.timeline.length) return;
    const intervalMs = Math.max(180, 900 / playSpeed);
    const id = setInterval(() => {
      setHourIdx((prev) => (prev + 1) % gridData.timeline.length);
    }, intervalMs);
    return () => clearInterval(id);
  }, [isPlaying, playSpeed, gridData]);

  // Atajos de teclado de la sala de operaciones
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      if (e.key === "1") setOpticMode("satellite");
      else if (e.key === "2") setOpticMode("flir");
      else if (e.key === "3") setOpticMode("wind");
      else if (e.key === "4") setOpticMode("risk");
      else if (e.key.toLowerCase() === "g") setCompareGfs((v) => !v);
      else if (e.key.toLowerCase() === "h") setHudVisible((v) => !v);
      else if (e.key.toLowerCase() === "r" || e.key === "Escape") {
        setResetSeq((s) => s + 1);
      } else if (e.code === "Space") {
        e.preventDefault();
        setIsPlaying((p) => !p);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const handleSimulateFire = async (lat: number, lon: number, label?: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/fire-plume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat,
          lon,
          hour_idx: hourIdx,
          duration_hours: 8,
          forecast_hours: forecastHours,
          start_date: historicalDate || null,
          end_date: historicalDate || null,
        }),
      });
      if (res.ok) {
        setFirePlume(await res.json());
        setOpticMode("risk");
        setFlyToTarget({ lat, lon, zoom: 10.4, seq: Date.now() });
        runGeometryAnalysis("pin", [[lat, lon]], label || `Foco (${lat.toFixed(2)}, ${lon.toFixed(2)})`);
      }
    } catch {
      // ignore
    }
  };

  const handleMapClick = (lat: number, lon: number) => {
    if (drawMode === "fire") {
      handleSimulateFire(lat, lon);
      return;
    }
    if (drawMode === "pin" || drawMode === "none") {
      const coords = [[lat, lon]];
      setDraftCoords(coords);
      runGeometryAnalysis("pin", coords, `Punto 3 km (${lat.toFixed(2)}, ${lon.toFixed(2)})`);
      return;
    }
    setDraftCoords((prev) => [...prev, [lat, lon]]);
  };

  const handleToggleVoice = () => {
    if (typeof window === "undefined") return;
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) {
      setVoiceTranscript("Tu navegador no soporta Web Speech API (prueba Chrome/Edge).");
      setTimeout(() => setVoiceTranscript(null), 3500);
      return;
    }
    if (voiceListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setVoiceListening(false);
      return;
    }

    const rec = new SpeechRec();
    rec.lang = "es-PE";
    rec.interimResults = false;
    rec.onresult = (ev: any) => {
      const cmd: string = ev.results?.[0]?.[0]?.transcript?.toLowerCase() || "";
      setVoiceTranscript(cmd);
      setTimeout(() => setVoiceTranscript(null), 4000);

      if (cmd.includes("huaraz")) {
        setFlyToTarget({ lat: -9.528, lon: -77.528, zoom: 10.6, seq: Date.now() });
        runGeometryAnalysis("pin", [[-9.528, -77.528]], "Huaraz · COER Áncash");
      } else if (cmd.includes("caraz")) {
        setFlyToTarget({ lat: -9.047, lon: -77.81, zoom: 10.8, seq: Date.now() });
        runGeometryAnalysis("pin", [[-9.047, -77.81]], "Caraz · Valle Agroexportador");
      } else if (cmd.includes("huascar")) {
        setFlyToTarget({ lat: -9.122, lon: -77.604, zoom: 11.2, seq: Date.now() });
        runGeometryAnalysis("pin", [[-9.122, -77.604]], "Nevado Huascarán");
      } else if (cmd.includes("pastoruri")) {
        setFlyToTarget({ lat: -9.927, lon: -77.193, zoom: 11.0, seq: Date.now() });
        runGeometryAnalysis("pin", [[-9.927, -77.193]], "Glaciar Pastoruri");
      } else if (cmd.includes("térmico") || cmd.includes("flir") || cmd.includes("helada")) {
        setOpticMode("flir");
      } else if (cmd.includes("viento")) {
        setOpticMode("wind");
      } else if (cmd.includes("riesgo") || cmd.includes("incendio")) {
        setOpticMode("risk");
      } else if (cmd.includes("satélite")) {
        setOpticMode("satellite");
      } else if (cmd.includes("comparar")) {
        setCompareGfs((v) => !v);
      } else if (cmd.includes("reproducir") || cmd.includes("pausa")) {
        setIsPlaying((p) => !p);
      } else if (cmd.includes("reiniciar") || cmd.includes("reset")) {
        setResetSeq((s) => s + 1);
      }
    };
    rec.onerror = () => setVoiceListening(false);
    rec.onend = () => setVoiceListening(false);
    recognitionRef.current = rec;
    setVoiceListening(true);
    rec.start();
  };

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-[#050911]">
      <CordilleraMap3D
        gridData={gridData}
        hourIdx={hourIdx}
        opticMode={opticMode}
        compareGfs={compareGfs}
        drawMode={drawMode}
        draftCoords={draftCoords}
        infrastructure={infrastructure}
        hotspots={hotspots}
        firePlume={firePlume}
        flyToTarget={flyToTarget}
        resetSeq={resetSeq}
        onMapClick={handleMapClick}
        onSelectHotspot={(spot) => handleSimulateFire(spot.lat, spot.lon, spot.nombre)}
        onSelectInfrastructure={(nombre, lat, lon) => {
          setDraftCoords([[lat, lon]]);
          runGeometryAnalysis("pin", [[lat, lon]], nombre);
        }}
      />

      <TacticalHUD
        gridData={gridData}
        opticMode={opticMode}
        compareGfs={compareGfs}
        drawMode={drawMode}
        draftCount={draftCoords.length}
        hudVisible={hudVisible}
        voiceListening={voiceListening}
        voiceTranscript={voiceTranscript}
        onChangeOptic={setOpticMode}
        onToggleCompareGfs={() => setCompareGfs((v) => !v)}
        onChangeDrawMode={(m) => {
          setDrawMode(m);
          if (m === "polygon" || m === "route") setDraftCoords([]);
        }}
        onExecuteDraft={() => {
          if (drawMode === "polygon" && draftCoords.length >= 3) {
            runGeometryAnalysis("polygon", draftCoords, "Polígono Agrícola / Distrito");
          } else if (drawMode === "route" && draftCoords.length >= 2) {
            runGeometryAnalysis("route", draftCoords, "Ruta Cordillera Blanca");
          }
        }}
        onClearDraft={() => {
          setDraftCoords([]);
          setFirePlume(null);
        }}
        onToggleHud={() => setHudVisible((v) => !v)}
        onResetView={() => setResetSeq((s) => s + 1)}
        onToggleVoice={handleToggleVoice}
      />

      {apiError && (
        <div className="pointer-events-auto absolute top-24 left-1/2 z-30 -translate-x-1/2 rounded-lg border border-rose-500 bg-rose-950/95 px-4 py-2 text-xs text-rose-200 shadow-xl">
          {apiError}
        </div>
      )}

      <DecisionSidebar
        analysis={analysis}
        briefing={briefing}
        hotspots={hotspots}
        firesBadge={firesBadge}
        firePlume={firePlume}
        hudVisible={hudVisible}
        loadingAnalysis={loadingAnalysis}
        onSelectHotspot={(spot) => handleSimulateFire(spot.lat, spot.lon, spot.nombre)}
        onFlyToPreset={(nombre, lat, lon) => {
          setFlyToTarget({ lat, lon, zoom: 10.6, seq: Date.now() });
          setDraftCoords([[lat, lon]]);
          runGeometryAnalysis("pin", [[lat, lon]], nombre);
        }}
      />

      <TimelineScrubber
        timeline={gridData?.timeline || []}
        hourIdx={hourIdx}
        isPlaying={isPlaying}
        playSpeed={playSpeed}
        forecastHours={forecastHours}
        historicalDate={historicalDate}
        hudVisible={hudVisible}
        onChangeHour={setHourIdx}
        onTogglePlay={() => setIsPlaying((p) => !p)}
        onChangeSpeed={setPlaySpeed}
        onChangeHorizon={(h) => {
          setHistoricalDate("");
          setForecastHours(h);
        }}
        onLoadHistorical={(d) => setHistoricalDate(d)}
      />
    </main>
  );
}
