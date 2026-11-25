"use client";

import React, { useEffect, useRef, useState } from "react";
import type {
  DrawMode,
  FireHotspot,
  FirePlumeResponse,
  GridForecastResponse,
  OpticMode,
} from "../types/inti3k";

interface CordilleraMap3DProps {
  gridData: GridForecastResponse | null;
  hourIdx: number;
  opticMode: OpticMode;
  compareGfs: boolean;
  drawMode: DrawMode;
  draftCoords: number[][];
  infrastructure: GeoJSON.FeatureCollection | null;
  hotspots: FireHotspot[];
  firePlume: FirePlumeResponse | null;
  flyToTarget: { lat: number; lon: number; zoom?: number; seq: number } | null;
  resetSeq: number;
  onMapClick: (lat: number, lon: number) => void;
  onSelectHotspot: (spot: FireHotspot) => void;
  onSelectInfrastructure: (nombre: string, lat: number, lon: number) => void;
}

function colorForTempFLIR(t: number): [number, number, number, number] {
  // Paleta térmica de alto contraste con foco en la isoterma de helada (0 °C)
  if (t <= -4) return [224, 242, 254, 215]; // Hielo extremo blanco-cian
  if (t <= 0) return [56, 189, 248, 210]; // Helada (< 0 °C) cian eléctrico
  if (t <= 3) return [59, 130, 246, 175]; // Riesgo helada (0..3 °C) azul
  if (t <= 8) return [16, 185, 129, 145]; // Frío andino verde esmeralda
  if (t <= 14) return [250, 204, 21, 150]; // Templado amarillo
  if (t <= 20) return [249, 115, 22, 175]; // Cálido naranja
  return [239, 68, 68, 195]; // Muy cálido rojo
}

function colorForWind(w: number): [number, number, number, number] {
  if (w < 2.5) return [15, 118, 110, 95];
  if (w < 5.0) return [20, 184, 166, 145];
  if (w < 8.0) return [250, 204, 21, 175];
  if (w < 11.0) return [249, 115, 22, 195];
  return [239, 68, 68, 215];
}

function colorForRisk(t: number, fireIdx: number): [number, number, number, number] {
  if (t <= 0 && fireIdx >= 55) return [217, 70, 239, 215]; // Doble riesgo magenta
  if (t <= 0) return [56, 189, 248, 205]; // Helada activa azul/cian
  if (fireIdx >= 65) return [239, 68, 68, 210]; // Incendio crítico rojo
  if (fireIdx >= 48) return [249, 115, 22, 170]; // Incendio moderado naranja
  if (t <= 2.5) return [59, 130, 246, 120]; // Precaución térmica
  return [15, 23, 42, 35]; // Zona segura casi transparente
}

function renderGridOverlayToDataUrl(
  gridData: GridForecastResponse,
  hourIdx: number,
  opticMode: OpticMode,
  compareGfs: boolean
): string {
  const rows = gridData.wrf_lats.length; // 70 (fila 0 = SUR)
  const cols = gridData.wrf_lons.length; // 35 (col 0 = OESTE)
  const scale = 6; // 210 x 420 px para dibujo nítido con vectores de viento
  const canvas = document.createElement("canvas");
  canvas.width = cols * scale;
  canvas.height = rows * scale;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const h = Math.min(hourIdx, gridData.frames.T_c.length - 1);
  const T = compareGfs ? gridData.frames.T_c_gfs[h] : gridData.frames.T_c[h];
  const W = compareGfs ? gridData.frames.W_ms_gfs[h] : gridData.frames.W_ms[h];
  const U = gridData.frames.U[h];
  const V = gridData.frames.V[h];
  const F = gridData.frames.fire_idx[h];

  for (let y = 0; y < rows; y++) {
    // INVERSIÓN CRÍTICA: en WRF fila 0 = SUR, en Canvas y=0 = NORTE (arriba)
    const r = rows - 1 - y;
    for (let x = 0; x < cols; x++) {
      let rgba: [number, number, number, number] = [0, 0, 0, 0];
      if (opticMode === "flir") {
        rgba = colorForTempFLIR(T[r][x]);
      } else if (opticMode === "wind") {
        rgba = colorForWind(W[r][x]);
      } else if (opticMode === "risk") {
        rgba = colorForRisk(T[r][x], F[r][x]);
      }
      ctx.fillStyle = `rgba(${rgba[0]},${rgba[1]},${rgba[2]},${rgba[3] / 255})`;
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  }

  // Dibujar vectores de dirección del viento (U, V) en modo Wind
  if (opticMode === "wind") {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
    ctx.lineWidth = 1.25;
    const step = 2;
    for (let y = 1; y < rows; y += step) {
      const r = rows - 1 - y;
      for (let x = 1; x < cols; x += step) {
        const u = U[r][x];
        const v = V[r][x];
        const mag = Math.hypot(u, v);
        if (mag < 0.4) continue;
        const cx = (x + 0.5) * scale;
        const cy = (y + 0.5) * scale;
        const len = Math.min(12, 3 + mag * 1.1);
        const dx = (u / mag) * len;
        const dy = -(v / mag) * len; // V positivo = hacia el Norte (arriba en canvas)
        ctx.beginPath();
        ctx.moveTo(cx - dx * 0.4, cy - dy * 0.4);
        ctx.lineTo(cx + dx * 0.6, cy + dy * 0.6);
        ctx.stroke();

        // Punta de flecha pequeña
        const angle = Math.atan2(dy, dx);
        ctx.beginPath();
        ctx.moveTo(cx + dx * 0.6, cy + dy * 0.6);
        ctx.lineTo(
          cx + dx * 0.6 - 3.2 * Math.cos(angle - Math.PI / 6),
          cy + dy * 0.6 - 3.2 * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(cx + dx * 0.6, cy + dy * 0.6);
        ctx.lineTo(
          cx + dx * 0.6 - 3.2 * Math.cos(angle + Math.PI / 6),
          cy + dy * 0.6 - 3.2 * Math.sin(angle + Math.PI / 6)
        );
        ctx.stroke();
      }
    }
  }

  return canvas.toDataURL("image/png");
}

export default function CordilleraMap3D({
  gridData,
  hourIdx,
  opticMode,
  compareGfs,
  drawMode,
  draftCoords,
  infrastructure,
  hotspots,
  firePlume,
  flyToTarget,
  resetSeq,
  onMapClick,
  onSelectHotspot,
  onSelectInfrastructure,
}: CordilleraMap3DProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<any>(null);
  const [mapReady, setMapReady] = useState(false);

  const callbacksRef = useRef({
    onMapClick,
    onSelectHotspot,
    onSelectInfrastructure,
  });
  callbacksRef.current = { onMapClick, onSelectHotspot, onSelectInfrastructure };

  // Inicializar MapLibre GL 3D Keyless (Esri World Imagery + AWS Terrarium DEM)
  useEffect(() => {
    let cancelled = false;
    let mapInstance: any = null;

    const initMap = async () => {
      if (!containerRef.current) return;
      const maplibregl = (await import("maplibre-gl")).default;
      if (cancelled || !containerRef.current) return;

      const style: any = {
        version: 8,
        sources: {
          esriSatellite: {
            type: "raster",
            tiles: [
              "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            ],
            tileSize: 256,
            maxzoom: 17,
            attribution: "Esri World Imagery",
          },
          terrainDem: {
            type: "raster-dem",
            tiles: [
              "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
            ],
            tileSize: 256,
            encoding: "terrarium",
            maxzoom: 14,
          },
        },
        terrain: {
          source: "terrainDem",
          exaggeration: 1.8,
        },
        layers: [
          {
            id: "background",
            type: "background",
            paint: { "background-color": "#050911" },
          },
          {
            id: "esri-satellite-layer",
            type: "raster",
            source: "esriSatellite",
            paint: {
              "raster-brightness-max": 0.9,
              "raster-contrast": 0.15,
            },
          },
        ],
      };

      mapInstance = new maplibregl.Map({
        container: containerRef.current,
        style,
        center: [-77.52, -9.35],
        zoom: 8.8,
        pitch: 52,
        bearing: -22,
        maxPitch: 75,
        antialias: true,
      });

      mapInstance.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");

      mapInstance.on("error", () => {
        // Silenciar errores si no hay conexión a tiles externos (el overlay de modelo 3 km sigue activo)
      });

      mapInstance.on("load", () => {
        if (cancelled) return;
        // Fuentes GeoJSON vacías iniciales
        mapInstance.addSource("infra-src", {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
        mapInstance.addLayer({
          id: "infra-circles",
          type: "circle",
          source: "infra-src",
          paint: {
            "circle-radius": 6,
            "circle-color": [
              "match",
              ["get", "categoria"],
              "laguna_presa",
              "#38bdf8",
              "montana_refugio",
              "#e2e8f0",
              "infraestructura",
              "#facc15",
              "#34d399",
            ],
            "circle-stroke-width": 2,
            "circle-stroke-color": "#050911",
          },
        });
        mapInstance.addLayer({
          id: "infra-labels",
          type: "symbol",
          source: "infra-src",
          layout: {
            "text-field": ["get", "nombre"],
            "text-size": 11,
            "text-offset": [0, 1.35],
            "text-anchor": "top",
          },
          paint: {
            "text-color": "#f8fafc",
            "text-halo-color": "#050911",
            "text-halo-width": 1.6,
          },
        });

        // Focos de incendio y cono de humo
        mapInstance.addSource("plume-src", {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
        mapInstance.addLayer({
          id: "plume-fill",
          type: "fill",
          source: "plume-src",
          paint: {
            "fill-color": "#f97316",
            "fill-opacity": 0.42,
          },
        });
        mapInstance.addLayer({
          id: "plume-outline",
          type: "line",
          source: "plume-src",
          paint: {
            "line-color": "#ef4444",
            "line-width": 2.2,
          },
        });

        mapInstance.addSource("fires-src", {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
        mapInstance.addLayer({
          id: "fires-circles",
          type: "circle",
          source: "fires-src",
          paint: {
            "circle-radius": 8,
            "circle-color": "#ef4444",
            "circle-stroke-width": 2.5,
            "circle-stroke-color": "#fef08a",
          },
        });

        // Geometrías dibujadas por el usuario (Pin, Polígono, Ruta)
        mapInstance.addSource("draw-src", {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
        mapInstance.addLayer({
          id: "draw-poly-fill",
          type: "fill",
          source: "draw-src",
          filter: ["==", "$type", "Polygon"],
          paint: {
            "fill-color": "#38bdf8",
            "fill-opacity": 0.22,
          },
        });
        mapInstance.addLayer({
          id: "draw-line",
          type: "line",
          source: "draw-src",
          paint: {
            "line-color": "#38bdf8",
            "line-width": 3,
          },
        });
        mapInstance.addLayer({
          id: "draw-pts",
          type: "circle",
          source: "draw-src",
          filter: ["==", "$type", "Point"],
          paint: {
            "circle-radius": 6.5,
            "circle-color": "#38bdf8",
            "circle-stroke-width": 2,
            "circle-stroke-color": "#ffffff",
          },
        });

        // Eventos clic en infraestructura y focos
        mapInstance.on("click", "infra-circles", (e: any) => {
          const feat = e.features?.[0];
          if (!feat) return;
          e.originalEvent.cancelBubble = true;
          const [lon, lat] = feat.geometry.coordinates;
          new maplibregl.Popup({ closeButton: false })
            .setLngLat([lon, lat])
            .setHTML(
              `<div style="font-weight:700;color:#38bdf8">${feat.properties.nombre}</div>` +
                `<div style="color:#cbd5e1;margin-top:2px">Elevación: ${feat.properties.elevacion_m} m · ${feat.properties.distrito}</div>` +
                `<div style="color:#94a3b8;margin-top:4px">${feat.properties.relevancia}</div>`
            )
            .addTo(mapInstance);
          callbacksRef.current.onSelectInfrastructure(feat.properties.nombre, lat, lon);
        });

        mapInstance.on("click", "fires-circles", (e: any) => {
          const feat = e.features?.[0];
          if (!feat) return;
          e.originalEvent.cancelBubble = true;
          const [lon, lat] = feat.geometry.coordinates;
          callbacksRef.current.onSelectHotspot({
            id: feat.properties.id,
            nombre: feat.properties.nombre,
            lat,
            lon,
            frp_mw: Number(feat.properties.frp_mw),
            confidence: feat.properties.confidence,
            source: feat.properties.source,
          });
        });

        mapInstance.on("mouseenter", "infra-circles", () => {
          mapInstance.getCanvas().style.cursor = "pointer";
        });
        mapInstance.on("mouseleave", "infra-circles", () => {
          mapInstance.getCanvas().style.cursor = "";
        });
        mapInstance.on("mouseenter", "fires-circles", () => {
          mapInstance.getCanvas().style.cursor = "pointer";
        });
        mapInstance.on("mouseleave", "fires-circles", () => {
          mapInstance.getCanvas().style.cursor = "";
        });

        mapInstance.on("click", (e: any) => {
          const hitInfra = mapInstance.queryRenderedFeatures(e.point, {
            layers: ["infra-circles", "fires-circles"],
          });
          if (hitInfra && hitInfra.length > 0) return;
          callbacksRef.current.onMapClick(e.lngLat.lat, e.lngLat.lng);
        });

        mapRef.current = mapInstance;
        setMapReady(true);
      });
    };

    initMap();
    return () => {
      cancelled = true;
      if (mapInstance) {
        mapInstance.remove();
      }
      mapRef.current = null;
      setMapReady(false);
    };
  }, []);

  // Actualizar cursor según herramienta de dibujo
  useEffect(() => {
    if (!mapReady || !mapRef.current) return;
    mapRef.current.getCanvas().style.cursor = drawMode !== "none" ? "crosshair" : "";
  }, [drawMode, mapReady]);

  // Actualizar capa Canvas 70x35 del modelo cuando cambie hora, óptica o toggle GFS vs IA
  useEffect(() => {
    if (!mapReady || !mapRef.current || !gridData) return;
    const map = mapRef.current;

    if (opticMode === "satellite") {
      if (map.getLayer("climate-overlay-layer")) {
        map.setLayoutProperty("climate-overlay-layer", "visibility", "none");
      }
      return;
    }

    const dataUrl = renderGridOverlayToDataUrl(gridData, hourIdx, opticMode, compareGfs);
    if (!dataUrl) return;

    const latMin = Math.min(...gridData.wrf_lats);
    const latMax = Math.max(...gridData.wrf_lats);
    const lonMin = Math.min(...gridData.wrf_lons);
    const lonMax = Math.max(...gridData.wrf_lons);

    const coords: [[number, number], [number, number], [number, number], [number, number]] = [
      [lonMin, latMax], // Top-Left (NW)
      [lonMax, latMax], // Top-Right (NE)
      [lonMax, latMin], // Bottom-Right (SE)
      [lonMin, latMin], // Bottom-Left (SW)
    ];

    const src = map.getSource("climate-overlay-src");
    if (src) {
      src.updateImage({ url: dataUrl, coordinates: coords });
      if (map.getLayer("climate-overlay-layer")) {
        map.setLayoutProperty("climate-overlay-layer", "visibility", "visible");
      }
    } else {
      map.addSource("climate-overlay-src", {
        type: "image",
        url: dataUrl,
        coordinates: coords,
      });
      const beforeId = map.getLayer("infra-circles") ? "infra-circles" : undefined;
      map.addLayer(
        {
          id: "climate-overlay-layer",
          type: "raster",
          source: "climate-overlay-src",
          paint: {
            "raster-opacity": 0.78,
            "raster-fade-duration": 0,
            "raster-resampling": compareGfs ? "linear" : "nearest",
          },
        },
        beforeId
      );
    }
  }, [mapReady, gridData, hourIdx, opticMode, compareGfs]);

  // Actualizar Infraestructura
  useEffect(() => {
    if (!mapReady || !mapRef.current || !infrastructure) return;
    const src = mapRef.current.getSource("infra-src");
    if (src) src.setData(infrastructure);
  }, [mapReady, infrastructure]);

  // Actualizar Focos de Incendio
  useEffect(() => {
    if (!mapReady || !mapRef.current) return;
    const src = mapRef.current.getSource("fires-src");
    if (!src) return;
    const fc: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: hotspots.map((h) => ({
        type: "Feature",
        properties: { ...h },
        geometry: { type: "Point", coordinates: [h.lon, h.lat] },
      })),
    };
    src.setData(fc);
  }, [mapReady, hotspots]);

  // Actualizar Cono de Humo 3 km
  useEffect(() => {
    if (!mapReady || !mapRef.current) return;
    const src = mapRef.current.getSource("plume-src");
    if (!src) return;
    if (!firePlume) {
      src.setData({ type: "FeatureCollection", features: [] });
      return;
    }
    const features: GeoJSON.Feature[] = [firePlume.cone_geojson];
    if (firePlume.trajectory.length >= 2) {
      features.push({
        type: "Feature",
        properties: {},
        geometry: {
          type: "LineString",
          coordinates: firePlume.trajectory.map((p) => [p.lon, p.lat]),
        },
      });
    }
    src.setData({ type: "FeatureCollection", features });
  }, [mapReady, firePlume]);

  // Actualizar Geometría Dibujada (Pin / Polígono / Ruta)
  useEffect(() => {
    if (!mapReady || !mapRef.current) return;
    const src = mapRef.current.getSource("draw-src");
    if (!src) return;

    const features: GeoJSON.Feature[] = draftCoords.map((pt) => ({
      type: "Feature",
      properties: {},
      geometry: { type: "Point", coordinates: [pt[1], pt[0]] },
    }));

    if (drawMode === "polygon" && draftCoords.length >= 3) {
      const ring = [...draftCoords.map((p) => [p[1], p[0]]), [draftCoords[0][1], draftCoords[0][0]]];
      features.push({
        type: "Feature",
        properties: {},
        geometry: { type: "Polygon", coordinates: [ring] },
      });
    } else if (draftCoords.length >= 2) {
      features.push({
        type: "Feature",
        properties: {},
        geometry: {
          type: "LineString",
          coordinates: draftCoords.map((p) => [p[1], p[0]]),
        },
      });
    }

    src.setData({ type: "FeatureCollection", features });
  }, [mapReady, draftCoords, drawMode]);

  // Vuelos de cámara (FlyTo / Reset)
  useEffect(() => {
    if (!mapReady || !mapRef.current || !flyToTarget) return;
    mapRef.current.flyTo({
      center: [flyToTarget.lon, flyToTarget.lat],
      zoom: flyToTarget.zoom ?? 10.6,
      pitch: 58,
      bearing: -18,
      duration: 1300,
    });
  }, [flyToTarget, mapReady]);

  useEffect(() => {
    if (!mapReady || !mapRef.current || resetSeq === 0) return;
    mapRef.current.flyTo({
      center: [-77.52, -9.35],
      zoom: 8.8,
      pitch: 52,
      bearing: -22,
      duration: 1100,
    });
  }, [resetSeq, mapReady]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      {/* Leyenda flotante compacta abajo-izquierda cuando hay capa climática activa */}
      {opticMode !== "satellite" && (
        <div className="pointer-events-none absolute bottom-24 left-4 z-10 rounded-lg border border-sky-500/25 bg-slate-950/85 px-3 py-2 text-[11px] backdrop-blur-md">
          <div className="font-semibold text-sky-300 mb-1">
            {opticMode === "flir" && (compareGfs ? "FLIR GFS 27 km (Borroso)" : "FLIR INTI-3K 3 km (Nítido)")}
            {opticMode === "wind" && (compareGfs ? "Viento GFS 27 km" : "Viento Encajonado 3 km (U,V)")}
            {opticMode === "risk" && "Riesgo Compuesto 3 km"}
          </div>
          {opticMode === "flir" && (
            <div className="flex items-center gap-2 text-slate-300">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-sky-400" /> ≤0°C Helada
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-500" /> 8°C
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-amber-400" /> 14°C
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-red-500" /> &gt;20°C
            </div>
          )}
          {opticMode === "wind" && (
            <div className="flex items-center gap-2 text-slate-300">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-teal-500" /> Suave
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-yellow-400" /> 5-8 m/s
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-red-500" /> &gt;11 m/s
            </div>
          )}
          {opticMode === "risk" && (
            <div className="flex items-center gap-2 text-slate-300">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-sky-400" /> Helada
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-orange-500" /> Incendio
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-fuchsia-500" /> Ambos
            </div>
          )}
        </div>
      )}
    </div>
  );
}
