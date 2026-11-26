"use client";

import React, { useEffect, useState } from "react";
import { Calendar, Pause, Play } from "lucide-react";
import type { HistoricalCase, TimelinePoint } from "../types/inti3k";

interface TimelineScrubberProps {
  timeline: TimelinePoint[];
  hourIdx: number;
  isPlaying: boolean;
  playSpeed: number;
  forecastHours: number;
  historicalDate: string;
  historicalCases: HistoricalCase[];
  hudVisible: boolean;
  onChangeHour: (idx: number) => void;
  onTogglePlay: () => void;
  onChangeSpeed: (speed: number) => void;
  onChangeHorizon: (hours: number) => void;
  onLoadHistorical: (dateStr: string) => void;
}

export default function TimelineScrubber({
  timeline,
  hourIdx,
  isPlaying,
  playSpeed,
  forecastHours,
  historicalDate,
  historicalCases,
  hudVisible,
  onChangeHour,
  onTogglePlay,
  onChangeSpeed,
  onChangeHorizon,
  onLoadHistorical,
}: TimelineScrubberProps) {
  const [dateInput, setDateInput] = useState(historicalDate);
  useEffect(() => setDateInput(historicalDate), [historicalDate]);

  if (!hudVisible || timeline.length === 0) return null;

  // Fechas pasadas: hasta anteayer (lo reciente se ve con el pronóstico actual)
  const maxDate = new Date(Date.now() - 2 * 86400000).toISOString().slice(0, 10);

  const current = timeline[Math.min(hourIdx, timeline.length - 1)];

  return (
    <div className="pointer-events-auto absolute bottom-4 left-4 right-4 z-20 md:right-[410px]">
      <div className="rounded-xl border border-sky-500/25 bg-slate-950/88 px-4 py-2.5 shadow-2xl backdrop-blur-md">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <button
              onClick={onTogglePlay}
              className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-600 text-white hover:bg-sky-500"
              title="Reproducir / Pausar secuencia temporal [Espacio]"
            >
              {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </button>
            <div className="flex items-center rounded-lg bg-slate-900 p-0.5 text-[11px]">
              {[1, 2, 4].map((sp) => (
                <button
                  key={sp}
                  onClick={() => onChangeSpeed(sp)}
                  className={`rounded-md px-1.5 py-0.5 font-semibold ${
                    playSpeed === sp ? "bg-sky-600 text-white" : "text-slate-400 hover:text-white"
                  }`}
                >
                  {sp}×
                </button>
              ))}
            </div>
            <div className="text-xs font-bold text-slate-100">
              {current?.time_lima} <span className="text-[10px] font-normal text-slate-400">(hora Perú, UTC−5)</span>
              {historicalDate && (
                <span className="ml-1.5 rounded bg-violet-900/70 px-1.5 py-0.5 text-[10px] font-semibold text-violet-200">
                  FECHA PASADA {historicalDate}
                </span>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span className="rounded bg-sky-950/80 px-2 py-0.5 text-sky-300">
              T mín dominio: <strong>{current?.tmin_c}°C</strong>
            </span>
            <span className="rounded bg-amber-950/70 px-2 py-0.5 text-amber-300">
              Viento máx: <strong>{current?.wmax_ms} m/s</strong>
            </span>
            <span className="rounded bg-slate-800 px-2 py-0.5 text-slate-300">
              Helada: <strong>{current?.frost_pct}%</strong>
            </span>

            <div className="flex items-center rounded-lg bg-slate-900 p-0.5 text-[11px]">
              {[24, 48].map((h) => (
                <button
                  key={h}
                  onClick={() => onChangeHorizon(h)}
                  className={`rounded-md px-2 py-0.5 font-medium ${
                    !historicalDate && forecastHours === h ? "bg-sky-600 text-white" : "text-slate-400"
                  }`}
                >
                  {h}h
                </button>
              ))}
            </div>

            {historicalCases.length > 0 && (
              <select
                value={historicalCases.find((c) => c.start_date === historicalDate)?.id ?? ""}
                onChange={(e) => {
                  const c = historicalCases.find((x) => x.id === e.target.value);
                  if (c) onLoadHistorical(c.start_date);
                }}
                className="rounded border border-violet-700/60 bg-slate-900 px-1.5 py-0.5 text-[11px] text-violet-200"
                title="Casos reales con GFS histórico (Open-Meteo)"
              >
                <option value="">Caso histórico…</option>
                {historicalCases.map((c) => (
                  <option key={c.id} value={c.id} title={c.nota}>
                    {c.label}
                  </option>
                ))}
              </select>
            )}

            <form
              onSubmit={(e) => {
                e.preventDefault();
                onLoadHistorical(dateInput);
              }}
              className="flex items-center gap-1"
            >
              <Calendar className="h-3.5 w-3.5 text-slate-400" />
              <input
                type="date"
                value={dateInput}
                max={maxDate}
                min="2021-03-23"
                onChange={(e) => setDateInput(e.target.value)}
                className="rounded border border-slate-700 bg-slate-900 px-1.5 py-0.5 text-[11px] text-slate-200"
                title="Fecha pasada (día en hora Perú): GFS histórico de Open-Meteo + modelo INTI-3K"
              />
              {dateInput && (
                <button
                  type="submit"
                  className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-sky-300 hover:bg-slate-700"
                >
                  Ir
                </button>
              )}
              {historicalDate && (
                <button
                  type="button"
                  onClick={() => {
                    setDateInput("");
                    onLoadHistorical("");
                  }}
                  className="rounded bg-rose-900/60 px-1.5 py-0.5 text-[10px] text-rose-200"
                >
                  Vivo
                </button>
              )}
            </form>
          </div>
        </div>

        <input
          type="range"
          min={0}
          max={Math.max(0, timeline.length - 1)}
          value={hourIdx}
          onChange={(e) => onChangeHour(Number(e.target.value))}
          className="h-1.5 w-full cursor-pointer accent-sky-400"
        />
      </div>
    </div>
  );
}
