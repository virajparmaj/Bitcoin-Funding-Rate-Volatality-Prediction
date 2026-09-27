import { useEffect, useId, useMemo, useRef, useState } from 'react';

export interface ChartPoint {
  label: string;
  x?: number;
  values: (number | null)[];
  low?: number | null;
  high?: number | null;
  count?: number;
  incomplete?: boolean;
  detail?: string;
}

interface ChartProps {
  points: ChartPoint[];
  names: string[];
  unit: string;
  label: string;
  compact?: boolean;
}

const HEIGHT = 270;
const LEFT = 62;
const RIGHT = 17;
const TOP = 16;
const BOTTOM = 40;

function display(value: number, unit: string) {
  if (unit === 'USD') return value.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (Math.abs(value) >= 1000) return value.toLocaleString('en-US', { maximumFractionDigits: 0 });
  return value.toFixed(3);
}

function linePath(points: ChartPoint[], column: number, x: (i: number) => number, y: (v: number) => number) {
  let previous = false;
  return points.map((point, i) => {
    const value = point.values[column];
    if (value === null || value === undefined) { previous = false; return ''; }
    const path = `${previous ? 'L' : 'M'}${x(i).toFixed(2)},${y(value).toFixed(2)}`;
    previous = true;
    return path;
  }).join('');
}

function bandPath(points: ChartPoint[], x: (i: number) => number, y: (v: number) => number) {
  const segments: number[][] = [];
  points.forEach((point, index) => {
    if (point.low == null || point.high == null) return;
    const last = segments.at(-1);
    if (last && last.at(-1) === index - 1) last.push(index);
    else segments.push([index]);
  });
  return segments.map(segment => {
    const upper = segment.map(i => `${x(i).toFixed(2)},${y(points[i].high!).toFixed(2)}`);
    const lower = [...segment].reverse().map(i => `${x(i).toFixed(2)},${y(points[i].low!).toFixed(2)}`);
    return `M${upper.join('L')}L${lower.join('L')}Z`;
  }).join('');
}

export function Chart({ points, names, unit, label, compact = false }: ChartProps) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1000);
  const [selection, setSelection] = useState<number | null>(null);
  const descriptionId = useId();
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(entries => {
      const measured = entries[0]?.contentRect.width;
      if (measured) setWidth(Math.max(300, Math.round(measured)));
    });
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const chart = useMemo(() => {
    const numbers = points.flatMap(p => [...p.values, p.low, p.high]).filter((v): v is number => v != null && Number.isFinite(v));
    if (!numbers.length) return null;
    let min = Math.min(...numbers);
    let max = Math.max(...numbers);
    const pad = (max - min) * .09 || Math.max(Math.abs(max) * .1, .1);
    min -= pad;
    max += pad;
    const xMin = points[0]?.x ?? 0;
    const xMax = points.at(-1)?.x ?? points.length - 1;
    const x = (i: number) => LEFT + (((points[i]?.x ?? i) - xMin) / (xMax - xMin || 1)) * (width - LEFT - RIGHT);
    const y = (v: number) => TOP + (1 - (v - min) / (max - min)) * (HEIGHT - TOP - BOTTOM);
    return { min, max, x, y, lines: names.map((_, i) => linePath(points, i, x, y)), band: bandPath(points, x, y) };
  }, [points, names, width]);

  if (!chart || !points.length) return <div className="chart-empty" role="status">No valid observations in this selection. Choose another range or variable.</div>;
  const selectedIndex = Math.min(selection ?? 0, points.length - 1);
  const selected = points[selectedIndex];
  const ticks = [0, .25, .5, .75, 1];
  const sampleIndices = width < 600 ? [...new Set([0, points.length - 1])] : [...new Set([0, Math.floor(points.length / 3), Math.floor(points.length * 2 / 3), points.length - 1])];

  return <div ref={container} className={`chart ${compact ? 'compact' : ''}`}>
    <div className="chart-key"><span className="mono unit-label">{unit}</span>{names.map((name, i) => <span className={`legend legend-${i}`} key={name}>{name}</span>)}{chart.band && <span className="legend band-legend">Daily min–max</span>}</div>
    <svg viewBox={`0 0 ${width} ${HEIGHT}`} aria-label={label} aria-describedby={descriptionId} role="img"
      onPointerMove={event => {
        const rect = event.currentTarget.getBoundingClientRect();
        const target = ((event.clientX - rect.left) / rect.width * width - LEFT) / (width - LEFT - RIGHT);
        setSelection(Math.max(0, Math.min(points.length - 1, Math.round(target * (points.length - 1)))));
      }} onPointerLeave={() => setSelection(null)}>
      <title>{label}</title>
      {ticks.map(t => {
        const value = chart.min + t * (chart.max - chart.min);
        return <g className="chart-grid" key={t}><line x1={LEFT} x2={width - RIGHT} y1={chart.y(value)} y2={chart.y(value)} /><text x={LEFT - 12} y={chart.y(value) + 4} textAnchor="end">{Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(0)}k` : value.toFixed(1)}</text></g>;
      })}
      {chart.min < 0 && chart.max > 0 && <line className="zero-line" x1={LEFT} x2={width - RIGHT} y1={chart.y(0)} y2={chart.y(0)} />}
      {chart.band && <path d={chart.band} className="range-band" />}
      {chart.lines.map((d, i) => <path d={d} key={i} className={`plot-line plot-line-${i}`} vectorEffect="non-scaling-stroke" />)}
      {points.length === 1 && points[0].values.map((value, i) => value === null ? null : <circle key={i} className={`plot-dot plot-dot-${i}`} r="3.5" cx={chart.x(0)} cy={chart.y(value)} />)}
      {points.map((p, i) => p.incomplete ? <line className="incomplete-mark" key={i} x1={chart.x(i)} x2={chart.x(i)} y1={HEIGHT - BOTTOM + 2} y2={HEIGHT - BOTTOM + 5} /> : null)}
      {sampleIndices.map((i, index) => <text key={i} className="x-label" x={chart.x(i)} y={HEIGHT - 10} textAnchor={index === 0 ? 'start' : index === sampleIndices.length - 1 ? 'end' : 'middle'}>{points[i].label}</text>)}
      {selection !== null && <g><line className="crosshair" x1={chart.x(selectedIndex)} x2={chart.x(selectedIndex)} y1={TOP} y2={HEIGHT - BOTTOM} />{selected.values.map((v, i) => v === null ? null : <circle key={i} className={`plot-dot plot-dot-${i}`} r="3.5" cx={chart.x(selectedIndex)} cy={chart.y(v)} />)}</g>}
    </svg>
    <div id={descriptionId} className="chart-readout" aria-live="polite"><span className="mono">{selected.label}</span>{selected.values.map((value, i) => <span key={names[i]}>{names[i]} <b className="mono">{value === null ? 'Unavailable' : `${display(value, unit)} ${unit}`}</b></span>)}{selected.low != null && selected.high != null && <span>Range <b className="mono">{display(selected.low, unit)}–{display(selected.high, unit)} {unit}</b></span>}{selected.count !== undefined && <span><b className="mono">n={selected.count}</b> valid observations</span>}{selected.detail && <span>{selected.detail}</span>}</div>
    <label className="chart-scrubber"><span>Inspect {points[0]?.count !== undefined ? 'day' : 'saved row'}</span><input type="range" min={0} max={points.length - 1} value={selectedIndex} onChange={event => setSelection(Number(event.target.value))} aria-label={`Inspect ${label}`} aria-valuetext={selected.label} /></label>
  </div>;
}
