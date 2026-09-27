export interface Source {
  id: string;
  path: string;
  locator: string;
  excerpt: string;
  sha256?: string;
  evidenceType?: string;
}

export interface Field {
  name: string;
  group: string;
  unit: string;
  missing_count: number;
  valid_count: number;
  missing_pct: number;
  definition: string;
}

export interface Metric {
  id: string;
  label: string;
  r2: number;
  mae_bp: number;
  rmse_bp: number;
  mse_native: number;
  n: number;
  status: string;
}

export interface Evidence {
  dataset: {
    row_count: number;
    column_count: number;
    arrival_start_utc: string;
    arrival_end_utc: string;
    funding_event_count: number;
    columns: Field[];
    preview: Record<string, string | number | null>[];
    caveats: string[];
  };
  rf: {
    n: number;
    metrics: Metric[];
    comparison: { excess_mse_pct: number; excess_mae_pct: number; skill_vs_current: number };
    constant_columns: Record<string, { distinct_count: number; values: number[] }>;
  };
  sarimax: { n: number; r2: number; scale_note: string };
  sources: Source[];
}

export interface DailyValue {
  count: number;
  missing_count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
}

export interface DailyRow {
  date: string;
  row_count: number;
  missing_day: boolean;
  funding_rate: DailyValue;
  mark_price: DailyValue;
  open_interest: DailyValue;
}

export interface PredictionRow {
  row_index: number;
  source_row_index: number;
  actual: number;
  predicted: number;
  current: number;
  ema3: number;
  lag1: number;
  residual: number;
}

export interface Series {
  raw_daily: DailyRow[];
  rf: PredictionRow[];
}

export interface Task {
  id: string;
  title: string;
  subtitle: string;
  target: string;
  units: string;
  inputs: string[];
  models: string[];
  split: string;
  preprocessing: string[];
  limitations: string[];
  status: string;
  sourceIds: string[];
}

export interface Feature {
  id: string;
  name: string;
  group: string;
  formula: string;
  purpose: string;
  engineeredIn: string;
  selectedBy: string[];
  savedIn: string[];
  limitations: string[];
  sourceIds: string[];
}

export interface StoredMetric {
  id: string;
  taskId: string;
  model: string;
  status: string;
  target: string;
  units: string;
  sampleCount: number | null;
  sampleCountNote: string;
  evaluation: string;
  values: Record<string, number>;
  sourceIds: string[];
}

export interface Research {
  tasks: Task[];
  features: Feature[];
  metrics: StoredMetric[];
  findings: { id: string; title: string; text: string; sourceIds: string[] }[];
  limitations: { id: string; title: string; text: string; sourceIds: string[] }[];
  plans: { id: string; title: string; text: string; status: string; sourceIds: string[] }[];
  corrections: { id: string; title: string; staleClaim: string; correction: string; sourceIds: string[] }[];
  historical: { id: string; taskId: string; title: string; status: string; text: string; evaluation: string; sourceIds: string[] }[];
  sources: Source[];
}
