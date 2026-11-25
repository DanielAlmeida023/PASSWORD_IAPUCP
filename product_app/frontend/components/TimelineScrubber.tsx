"use client";

import React, { useState } from "react";
import { Calendar, Pause, Play } from "lucide-react";
import type { TimelinePoint } from "../types/inti3k";

interface TimelineScrubberProps {
  timeline: TimelinePoint[];
  hourIdx: number;
  isPlaying: boolean;
  playSpeed: number;
  forecastHours: number;
  historicalDate: string;
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
  hudVisible,
  onChangeHour,
  onTogglePlay,
  onChangeSpeed,
  onChangeHorizon,
  onLoadHistorical,
}: TimelineScrubberProps) {
  const [dateInput, setDateInput] = useState(historicalDate);

  if (!hudVisible || timeline.length === 0) return null;

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
              {current?.time_lima} <span className="text-[10px] font-normal text-slate-400">(Hora Perú UTC−5)</span>
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
                onChange={(e) => setDateInput(e.target.value)}
                className="rounded border border-slate-700 bg-slate-900 px-1.5 py-0.5 text-[11px] text-slate-200"
                title="Consultar fecha pasada del GFS (opcional)"
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
