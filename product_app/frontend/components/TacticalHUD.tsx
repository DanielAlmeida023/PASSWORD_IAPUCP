"use client";

import React from "react";
import {
  Eye,
  EyeOff,
  Flame,
  Layers,
  MapPin,
  Mic,
  MicOff,
  Pentagon,
  RotateCcw,
  Route,
  Satellite,
  Thermometer,
  Trash2,
  Wind,
} from "lucide-react";
import type { DrawMode, GridForecastResponse, OpticMode } from "../types/inti3k";

interface TacticalHUDProps {
  gridData: GridForecastResponse | null;
  opticMode: OpticMode;
  compareGfs: boolean;
  drawMode: DrawMode;
  draftCount: number;
  hudVisible: boolean;
  voiceListening: boolean;
  voiceTranscript: string | null;
  onChangeOptic: (mode: OpticMode) => void;
  onToggleCompareGfs: () => void;
  onChangeDrawMode: (mode: DrawMode) => void;
  onExecuteDraft: () => void;
  onClearDraft: () => void;
  onToggleHud: () => void;
  onResetView: () => void;
  onToggleVoice: () => void;
}

export default function TacticalHUD({
  gridData,
  opticMode,
  compareGfs,
  drawMode,
  draftCount,
  hudVisible,
  voiceListening,
  voiceTranscript,
  onChangeOptic,
  onToggleCompareGfs,
  onChangeDrawMode,
  onExecuteDraft,
  onClearDraft,
  onToggleHud,
  onResetView,
  onToggleVoice,
}: TacticalHUDProps) {
  if (!hudVisible) {
    return (
      <div className="pointer-events-auto absolute top-3 right-3 z-20 flex items-center gap-2">
        <button
          onClick={onToggleHud}
          className="flex items-center gap-1.5 rounded-lg border border-sky-500/40 bg-slate-950/90 px-3 py-1.5 text-xs font-semibold text-sky-300 shadow-lg backdrop-blur-md hover:bg-slate-900"
          title="Mostrar HUD (Tecla H)"
        >
          <Eye className="h-3.5 w-3.5" /> Mostrar HUD [H]
        </button>
      </div>
    );
  }

  const model = gridData?.model;
  const diffMax = gridData?.domain_kpis?.max_gfs_vs_ai_diff_c;

  return (
    <header className="pointer-events-none absolute top-3 left-3 right-3 z-20 flex flex-col gap-2">
      <div className="pointer-events-auto flex flex-wrap items-center justify-between gap-2 rounded-xl border border-sky-500/25 bg-slate-950/88 px-3.5 py-2.5 shadow-2xl backdrop-blur-md">
        {/* Identidad y Transparencia Científica */}
        <div className="flex items-center gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs font-bold tracking-wider text-sky-400 uppercase">
                INTI-3K · COER ÁNCASH / CORDILLERA BLANCA
              </span>
              <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-medium text-slate-300">
                Equipo PASSWORD
              </span>
            </div>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[10px] text-slate-400">
              <span className="rounded border border-emerald-500/30 bg-emerald-950/40 px-1.5 py-0.5 text-emerald-300">
                Modelo Kaggle: {model?.val_skill_score ?? 0.3835} Skill
              </span>
              <span>
                Incertidumbre Val: T ±{model?.val_rmse?.T ?? 0.93}°C · RH ±{model?.val_rmse?.RH ?? 8.58}% · Viento ±
                {model?.val_rmse?.U ?? 1.62} m/s
              </span>
              {gridData?.source_mode === "offline_cache_fallback" && (
                <span className="rounded bg-amber-900/60 px-1.5 py-0.5 text-amber-300">
                  Caché Local Activo
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Selector de Ópticas [1-4] y Comparador GFS vs IA [G] */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            onClick={() => onChangeOptic("satellite")}
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
              opticMode === "satellite"
                ? "bg-sky-600 text-white shadow"
                : "bg-slate-900/90 text-slate-300 hover:bg-slate-800"
            }`}
          >
            <Satellite className="h-3.5 w-3.5" /> [1] Satélite
          </button>
          <button
            onClick={() => onChangeOptic("flir")}
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
              opticMode === "flir"
                ? "bg-sky-600 text-white shadow"
                : "bg-slate-900/90 text-slate-300 hover:bg-slate-800"
            }`}
          >
            <Thermometer className="h-3.5 w-3.5" /> [2] FLIR Térmico
          </button>
          <button
            onClick={() => onChangeOptic("wind")}
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
              opticMode === "wind"
                ? "bg-sky-600 text-white shadow"
                : "bg-slate-900/90 text-slate-300 hover:bg-slate-800"
            }`}
          >
            <Wind className="h-3.5 w-3.5" /> [3] Viento 3 km
          </button>
          <button
            onClick={() => onChangeOptic("risk")}
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
              opticMode === "risk"
                ? "bg-sky-600 text-white shadow"
                : "bg-slate-900/90 text-slate-300 hover:bg-slate-800"
            }`}
          >
            <Layers className="h-3.5 w-3.5" /> [4] Riesgo
          </button>

          <div className="mx-1 h-5 w-px bg-slate-700" />

          {/* Toggle GFS 27km vs IA 3km */}
          <button
            onClick={onToggleCompareGfs}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition ${
              compareGfs
                ? "border-amber-400/70 bg-amber-500/20 text-amber-200"
                : "border-emerald-400/60 bg-emerald-500/20 text-emerald-200"
            }`}
            title="Alternar entre Pronóstico Global GFS (27 km) y Modelo IA INTI-3K (3 km) [Tecla G]"
          >
            <span>[G] {compareGfs ? "Viendo: GFS 27 km" : "Viendo: INTI-3K 3 km"}</span>
            {diffMax !== undefined && (
              <span className="rounded bg-slate-950/70 px-1 text-[10px] text-sky-300">Δ {diffMax}°C</span>
            )}
          </button>
        </div>

        {/* Herramientas de Dibujo + Voz + Reset/HUD */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            onClick={() => onChangeDrawMode("pin")}
            className={`flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs transition ${
              drawMode === "pin" ? "bg-cyan-600 text-white" : "bg-slate-900 text-slate-300 hover:bg-slate-800"
            }`}
            title="Consultar un punto exacto de 3 km"
          >
            <MapPin className="h-3.5 w-3.5" /> Pin
          </button>
          <button
            onClick={() => onChangeDrawMode("polygon")}
            className={`flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs transition ${
              drawMode === "polygon" ? "bg-cyan-600 text-white" : "bg-slate-900 text-slate-300 hover:bg-slate-800"
            }`}
            title="Dibujar parcela agrícola o distrito"
          >
            <Pentagon className="h-3.5 w-3.5" /> Polígono
          </button>
          <button
            onClick={() => onChangeDrawMode("route")}
            className={`flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs transition ${
              drawMode === "route" ? "bg-cyan-600 text-white" : "bg-slate-900 text-slate-300 hover:bg-slate-800"
            }`}
            title="Trazar ruta de montaña A→B"
          >
            <Route className="h-3.5 w-3.5" /> Ruta
          </button>
          <button
            onClick={() => onChangeDrawMode("fire")}
            className={`flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs transition ${
              drawMode === "fire"
                ? "bg-orange-600 text-white"
                : "bg-slate-900 text-orange-300 hover:bg-slate-800"
            }`}
            title="Clic en cualquier cerro para simular foco de incendio y cono de humo 3 km"
          >
            <Flame className="h-3.5 w-3.5" /> Foco
          </button>

          {((drawMode === "polygon" && draftCount >= 3) || (drawMode === "route" && draftCount >= 2)) && (
            <button
              onClick={onExecuteDraft}
              className="rounded-lg bg-emerald-600 px-2.5 py-1.5 text-xs font-bold text-white shadow hover:bg-emerald-500"
            >
              Analizar ({draftCount} pts)
            </button>
          )}
          {draftCount > 0 && (
            <button
              onClick={onClearDraft}
              className="rounded-lg bg-slate-900 p-1.5 text-slate-400 hover:text-red-400"
              title="Limpiar geometría"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}

          <div className="mx-0.5 h-5 w-px bg-slate-700" />

          <button
            onClick={onToggleVoice}
            className={`flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs transition ${
              voiceListening
                ? "bg-rose-600 text-white animate-pulse"
                : "bg-slate-900 text-slate-300 hover:bg-slate-800"
            }`}
            title="Comando de voz (ej: 'Huaraz', 'Huascarán', 'térmico', 'viento', 'comparar')"
          >
            {voiceListening ? <Mic className="h-3.5 w-3.5" /> : <MicOff className="h-3.5 w-3.5" />}
          </button>
          <button
            onClick={onResetView}
            className="rounded-lg bg-slate-900 p-1.5 text-slate-300 hover:bg-slate-800"
            title="Reiniciar vista panorámica [R]"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onToggleHud}
            className="rounded-lg bg-slate-900 p-1.5 text-slate-300 hover:bg-slate-800"
            title="Ocultar HUD [H]"
          >
            <EyeOff className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Hint contextual de herramienta o comando de voz */}
      {(drawMode === "polygon" || drawMode === "route" || drawMode === "fire" || voiceTranscript) && (
        <div className="pointer-events-none flex justify-center">
          <div className="rounded-full border border-sky-500/30 bg-slate-950/90 px-4 py-1 text-[11px] text-sky-200 shadow-lg backdrop-blur-md">
            {voiceTranscript ? (
              <span>🎙️ Voz: &ldquo;{voiceTranscript}&rdquo;</span>
            ) : drawMode === "polygon" ? (
              <span>Haz clic en el mapa para marcar vértices de la parcela/distrito ({draftCount} puntos). Mínimo 3 puntos y pulsa &ldquo;Analizar&rdquo;.</span>
            ) : drawMode === "route" ? (
              <span>Marca los waypoints de tu ruta A→B ({draftCount} puntos). Mínimo 2 puntos y pulsa &ldquo;Analizar&rdquo;.</span>
            ) : (
              <span>Haz clic en cualquier montaña o quebrada para simular un foco de incendio y proyectar el cono de humo con viento 3 km.</span>
            )}
          </div>
        </div>
      )}
    </header>
  );
}
