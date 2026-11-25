export type OpticMode = "satellite" | "flir" | "wind" | "risk";
export type DrawMode = "pin" | "polygon" | "route" | "fire" | "none";

export interface ModelMetadata {
  loaded: boolean;
  exp_id?: string;
  arch: string;
  arch_label: string;
  val_skill_score: number;
  weights_path: string;
  weights_mb: number;
  val_rmse: Record<string, number>;
}

export interface TimelinePoint {
  idx: number;
  time_utc: string;
  time_lima: string;
  tmin_c: number;
  tmax_c: number;
  wmax_ms: number;
  frost_pct: number;
  high_fire_pct: number;
}

export interface GridForecastResponse {
  model: ModelMetadata;
  source_mode: string;
  inference_ms: number;
  periodo_lima: string;
  wrf_lats: number[];
  wrf_lons: number[];
  dem: number[][];
  timeline: TimelinePoint[];
  frames: {
    T_c: number[][][];
    RH: number[][][];
    U: number[][][];
    V: number[][][];
    W_ms: number[][][];
    fire_idx: number[][][];
    T_c_gfs: number[][][];
    W_ms_gfs: number[][][];
  };
  domain_kpis: {
    total_cells_3km: number;
    pct_helada_24h: number;
    pct_riesgo_incendio_alto: number;
    tmin_dominio_c: number;
    tmax_dominio_c: number;
    wmax_dominio_ms: number;
    max_gfs_vs_ai_diff_c: number;
  };
}

export interface GeometryAnalysisResponse {
  label: string;
  geometry_type: "pin" | "polygon" | "route";
  cells_3km_count: number;
  area_km2: number;
  route_distance_km: number | null;
  elevacion_min_m: number;
  elevacion_max_m: number;
  elevacion_media_m: number;
  kpis: {
    tmin_c: number;
    tmax_c: number;
    hora_tmin_lima: string;
    frost_hours: number;
    cells_with_frost: number;
    viento_max_ms: number;
    wind_chill_min_c: number;
    rh_min_pct: number;
    fire_risk_max: number;
    gfs_27km_tmin_c: number;
    delta_resolucion_c: number;
  };
  actionable_status: string;
  recommendation: string;
  hourly_series: Array<{
    idx: number;
    time_utc: string;
    time_lima: string;
    temp_min_c: number;
    temp_mean_c: number;
    temp_gfs_27km_c: number;
    diff_gfs_vs_ai_c: number;
    rh_min_pct: number;
    wind_max_ms: number;
    u_mean_ms: number;
    v_mean_ms: number;
    wind_chill_c: number;
    fire_risk: number;
  }>;
  route_profile: Array<{
    dist_km: number;
    lat: number;
    lon: number;
    elevacion_m: number;
    tmin_c: number;
    wmax_ms: number;
    wind_chill_min_c: number;
    fire_risk_max: number;
  }>;
  selected_cells_coords: number[][];
}

export interface FireHotspot {
  id: string;
  nombre: string;
  lat: number;
  lon: number;
  frp_mw: number;
  confidence: string;
  source: string;
}

export interface FiresFirmsResponse {
  mode: string;
  badge: string;
  count: number;
  hotspots: FireHotspot[];
}

export interface FirePlumeResponse {
  origin: { lat: number; lon: number };
  start_time_lima: string;
  duration_hours: number;
  avg_fire_risk: number;
  max_wind_ms: number;
  trajectory: Array<{
    step_h: number;
    time_lima: string;
    lat: number;
    lon: number;
    u_ms: number;
    v_ms: number;
    wind_ms: number;
    rh_pct: number;
    fire_risk: number;
  }>;
  cone_geojson: GeoJSON.Feature<GeoJSON.Polygon>;
}

export interface BriefingResponse {
  spanish: string;
  quechua_ancash: string;
  generator: string;
}
