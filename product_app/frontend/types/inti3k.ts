export type OpticMode = "satellite" | "flir" | "wind" | "risk";
export type DrawMode = "pin" | "polygon" | "route" | "fire" | "none";

export interface ModelMetadata {
  loaded: boolean;
  name?: string;
  exp_id?: string;
  commit?: string;
  arch: string;
  arch_label: string;
  val_skill_score: number | null;
  val_fold?: string;
  kaggle_public_lb?: number | null;
  weights_path: string;
  weights_mb: number;
  val_rmse: Record<string, number>;
}

/** Respuesta de /api/model-info (models/model_card.json + verificación de pesos servidos). */
export interface ModelInfo {
  name: string;
  short_name?: string;
  arch: string;
  commit: string;
  validation: {
    skill_score: number;
    fold: string;
    rmse_per_var: Record<string, number>;
    skill_per_var: Record<string, number>;
  };
  kaggle: { public_lb: number; public_lb_rank?: number };
  members: Array<{ name: string; width: number; params: number }>;
  total_params: number;
  weights: { path: string; size_bytes: number; size_mb: number; sha256: string };
  served_weights: { exists: boolean; size_mb?: number; sha256?: string; matches_card?: boolean };
}

export interface GfsStatus {
  mode: "open_meteo_live" | "cache_fresh" | "offline_cache_fallback" | "open_meteo_historical" | "historical_cache" | string;
  label: string;
  using_cache: boolean;
  is_fallback: boolean;
  fetched_at_utc: string | null;
  fetched_at_peru: string | null;
  age_h: number | null;
  error: string | null;
  refresh_every_h: number;
  file?: string;
}

export interface LagReport {
  lags_h: number[];
  n_output_hours: number;
  n_hours_all_lags_exact: number;
  n_lag_slots_near_fill: number;
  n_lag_slots_self_fill: number;
  padded_hours_utc: string[];
}

export interface HistoricalCase {
  id: string;
  label: string;
  start_date: string;
  end_date: string;
  nota: string;
}

export interface Cell3km {
  row: number;
  col: number;
  lat: number;
  lon: number;
  elevacion_m: number;
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
  gfs_status: GfsStatus;
  historical: boolean;
  lag_report: LagReport;
  tz_label: string;
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
    pct_helada_periodo: number;
    pct_helada_gfs_periodo: number;
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
  tz_label: string;
  source_mode: string;
  gfs_status: GfsStatus;
  historical: boolean;
  critical_cell: Cell3km & { hora_lima: string; time_utc: string };
  comparador: {
    cell: Cell3km;
    hora_lima: string;
    inti3k_c: number;
    gfs_27km_c: number;
    gfs_27km_min_celda_c: number;
    delta_c: number;
    texto: string;
    veredicto: string;
    metodo_gfs: string;
  };
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
    frost_cell_hours: number;
    frost_hours_worst_cell: number;
    frost_hours_gfs: number;
    viento_max_ms: number;
    wind_chill_min_c: number;
    rh_min_pct: number;
    fire_risk_max: number;
    gfs_27km_tmin_c: number;
    delta_resolucion_c: number;
  };
  actionable_status: string;
  alerta_principal_tipo: "helada";
  recommendation: string;
  alertas_secundarias: Array<{
    tipo: "incendio_sequedad" | "viento";
    nivel: "alta" | "moderada";
    titulo: string;
    valor: number;
    unidad: string;
    hora_lima: string;
    cell: Cell3km;
    texto: string;
    recomendacion: string;
  }>;
  hourly_series: Array<{
    idx: number;
    time_utc: string;
    time_lima: string;
    temp_min_c: number;
    temp_mean_c: number;
    temp_cell_c: number;
    temp_gfs_cell_c: number;
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
