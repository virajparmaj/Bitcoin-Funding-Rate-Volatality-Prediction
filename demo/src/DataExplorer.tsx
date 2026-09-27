import { useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import * as Tabs from '@radix-ui/react-tabs';
import type { Evidence, Feature, Research, Series } from './types';
import { Chart } from './components/Chart';
import { Note, SectionHeading, SourceLink, bp, number } from './components/ui';

const VARIABLES = {
  funding_rate: { label: 'Funding indication', unit: 'bp' },
  mark_price: { label: 'Mark price', unit: 'USDT / BTC' },
  open_interest: { label: 'Open interest', unit: 'provider units' },
  predicted_funding_rate: { label: 'Provider predicted funding rate', unit: 'Unavailable' },
};
type Variable = keyof typeof VARIABLES;

function Timeline({ evidence, series }: { evidence: Evidence; series: Series }) {
  const [variable, setVariable] = useState<Variable>('funding_rate');
  const [start, setStart] = useState(evidence.dataset.arrival_start_utc.slice(0, 10));
  const [end, setEnd] = useState(evidence.dataset.arrival_end_utc.slice(0, 10));
  const datesValid = Boolean(start && end && start <= end);
  const selected = datesValid ? series.raw_daily.filter(row => row.date >= start && row.date <= end) : [];
  const field = variable === 'predicted_funding_rate' ? null : variable;
  const points = selected.map(row => {
    const value = field ? row[field] : { mean: null, min: null, max: null, count: 0, missing_count: row.row_count };
    const convert = (v: number | null) => field === 'funding_rate' ? bp(v) : v;
    return { label: row.date, values: [convert(value.mean)], low: convert(value.min), high: convert(value.max), count: value.count, incomplete: row.row_count !== 24 || value.missing_count > 0, detail: `${value.missing_count} missing field values; ${row.row_count} arrival rows` };
  });
  const valid = field ? selected.reduce((sum, row) => sum + row[field].count, 0) : 0;
  const missing = selected.reduce((sum, row) => sum + (field ? row[field].missing_count : row.row_count), 0);
  const fieldInfo = evidence.dataset.columns.find(column => column.name === variable)!;
  return <>
    <section className="panel timeline-panel"><div className="panel-heading"><div><div className="eyebrow">RAW OBSERVATIONS / ARRIVAL TIME</div><h2>{VARIABLES[variable].label}</h2></div><SourceLink ids={['raw', 'synthetic-time', 'import-sampling']}>Data & clock</SourceLink></div>
      <div className="filter-bar">
        <label>Variable<select value={variable} onChange={event => setVariable(event.target.value as Variable)}>{Object.entries(VARIABLES).map(([key, value]) => <option value={key} key={key}>{value.label}</option>)}</select></label>
        <label>From · UTC<input type="date" value={start} min={evidence.dataset.arrival_start_utc.slice(0, 10)} max={evidence.dataset.arrival_end_utc.slice(0, 10)} onInput={event => setStart(event.currentTarget.value)} onChange={event => setStart(event.target.value)} /></label>
        <label>Through · UTC<input type="date" value={end} min={evidence.dataset.arrival_start_utc.slice(0, 10)} max={evidence.dataset.arrival_end_utc.slice(0, 10)} onInput={event => setEnd(event.currentTarget.value)} onChange={event => setEnd(event.target.value)} /></label>
        <button className="text-button reset-button" onClick={() => { setStart(evidence.dataset.arrival_start_utc.slice(0, 10)); setEnd(evidence.dataset.arrival_end_utc.slice(0, 10)); }}>Reset dates</button>
      </div>
      {!datesValid ? <div className="chart-empty" role="status">Choose a start and end date in chronological order, or reset dates.</div> : <Chart key={variable + start + end} points={points} names={['Daily mean']} unit={VARIABLES[variable].unit} label={`${VARIABLES[variable].label} by arrival date UTC`} />}
      {variable === 'predicted_funding_rate' ? <Note>Unavailable in every raw row. This provider field is distinct from this project’s saved predictions. The current <code>funding_rate</code> indication is available in many rows.</Note> : <div className="descriptive-stats"><span><b className="mono">{number(valid)}</b> valid values in selection</span><span><b className="mono">{number(missing)}</b> missing field values</span><span><b className="mono">{number(selected.length)}</b> calendar days</span></div>}
      <p className="chart-caption">Every UTC day is retained. Line = mean of valid observations; band = observed daily minimum–maximum, preserving spikes. Missing days are gaps. Ticks beneath the axis flag days with field missingness or an arrival-row count other than the nominal 24; they do not establish missing hourly messages. One raw row has no arrival time and cannot enter calendar charts. Date filters change this descriptive view only.</p>
    </section>
    <div className="data-notes"><div><h3>What this field represents</h3><p>{fieldInfo.definition}</p><span className="mono small">Units: {fieldInfo.unit}</span></div><div><h3>Two clocks, different purposes</h3><p>Valid <code>local_timestamp</code> values define these dates. The supplied <code>timestamp</code> is a synthetic row grid and is never presented as verified UTC.</p><SourceLink ids={['raw', 'synthetic-time']}>Inspect timestamp provenance</SourceLink></div><div><h3>Indications, not payment labels</h3><p>These are nominally hourly derivative-ticker observations. Funding event IDs do not independently verify settled rates or make the rows independent samples.</p><SourceLink ids={['raw']}>Raw dataset record</SourceLink></div></div>
  </>;
}

function RawFields({ evidence }: { evidence: Evidence }) {
  return <div className="panel"><div className="panel-heading"><div><div className="eyebrow">THE RAW DATA CONTRACT</div><h2>Availability before preprocessing</h2></div><SourceLink ids={['raw']}>CSV source</SourceLink></div><div className="table-scroll"><table className="field-table"><caption>{number(evidence.dataset.row_count)} rows. Missing values remain missing; no fill is applied to these counts.</caption><thead><tr><th>Field</th><th>Meaning</th><th className="numeric">Valid</th><th className="numeric">Missing</th><th>Missing share</th></tr></thead><tbody>{evidence.dataset.columns.map(field => <tr key={field.name}><td><code>{field.name}</code><small>{field.group} · {field.unit}</small></td><td>{field.definition}</td><td className="numeric">{number(field.valid_count)}</td><td className="numeric">{number(field.missing_count)}</td><td><div className="missing-meter"><span style={{ width: `${field.missing_pct}%` }} /></div><span className="mono small">{field.missing_pct.toFixed(2)}%</span></td></tr>)}</tbody></table></div>
    <details className="disclosure"><summary>Inspect a small raw-data preview <span className="mono">FIRST ROWS + MISSING-VALUE EXAMPLES</span></summary><p>Original units, before filling. Time fields are epoch microseconds; <code>timestamp</code> is synthetic. “Missing” denotes an absent raw value.</p><div className="table-scroll"><table className="raw-table"><thead><tr><th>Raw row</th>{evidence.dataset.columns.map(field => <th key={field.name}>{field.name}</th>)}</tr></thead><tbody>{evidence.dataset.preview.map(row => <tr key={row.row_index}><td className="mono">{row.row_index}</td>{evidence.dataset.columns.map(field => <td className="mono" key={field.name}>{row[field.name] === null ? <span className="missing-value">Missing</span> : String(row[field.name])}</td>)}</tr>)}</tbody></table></div></details>
  </div>;
}

function FeatureDetails({ feature }: { feature: Feature }) {
  return <details className="feature-row">
    <summary><code>{feature.name}</code><span>{feature.group}</span><span className="expand-symbol">+</span></summary>
    <div className="feature-expanded">
      <div className="feature-formula"><span className="eyebrow">EXECUTABLE DEFINITION</span><code>{feature.formula}</code><p>{feature.purpose} <span className="muted">Design rationale; a measured contribution is not established.</span></p></div>
      <dl className="feature-facts">
        <div><dt>Defined in</dt><dd>{feature.engineeredIn}</dd></div>
        <div><dt>Experiment selection</dt><dd>{feature.selectedBy.length ? feature.selectedBy.join(' · ') : 'Not explicitly named by the inspected selectors; see the conditional Model 3 selection note below.'}</dd></div>
        <div><dt>Present in saved files</dt><dd>{feature.savedIn.length ? feature.savedIn.join(' · ') : 'Not present in the inspected saved prediction headers.'}</dd></div>
      </dl>
      {feature.limitations.map(note => <p className="feature-caveat" key={note}>{note}</p>)}
      <SourceLink ids={feature.sourceIds}>Definition & experiment selectors</SourceLink>
    </div>
  </details>;
}

function Features({ research }: { research: Research }) {
  const [query, setQuery] = useState('');
  const [group, setGroup] = useState('All groups');
  const [model, setModel] = useState('All experiments');
  const groups = useMemo(() => [...new Set(research.features.map(feature => feature.group))], [research]);
  const models = useMemo(() => [...new Set(research.features.flatMap(feature => feature.selectedBy))].sort(), [research]);
  const features = research.features.filter(feature => (group === 'All groups' || feature.group === group) && (model === 'All experiments' || feature.selectedBy.includes(model)) && `${feature.name} ${feature.formula} ${feature.purpose}`.toLowerCase().includes(query.toLowerCase()));
  return <section className="panel"><div className="panel-heading"><div><div className="eyebrow">FEATURE DICTIONARY</div><h2>From observations to model inputs</h2></div><span className="mono small muted">{features.length} / {research.features.length} definitions</span></div><p className="panel-description">Inspect the formula, intended purpose, experiment selection, and saved-file presence separately. Lookbacks count rows or observations.</p><div className="filter-bar feature-filters"><label className="search-label"><span>Find a feature</span><span className="search-input"><Search size={14} /><input type="search" placeholder="Search names or formulas" value={query} onChange={event => setQuery(event.target.value)} /></span></label><label>Feature group<select value={group} onChange={event => setGroup(event.target.value)}><option>All groups</option>{groups.map(value => <option key={value}>{value}</option>)}</select></label><label>Selected by experiment<select value={model} onChange={event => setModel(event.target.value)}><option>All experiments</option>{models.map(value => <option key={value}>{value}</option>)}</select></label></div><Note>The current helper creates <code>volatility_5h</code> from five mark-price levels. Legacy notebooks and saved files use <code>volatility_5min</code>. The name mismatch does not establish five-minute data or the exact historical formula.</Note><div className="feature-list">{features.map(feature => <FeatureDetails key={feature.id} feature={feature} />)}{features.length === 0 && <div className="chart-empty" role="status">No features match these filters. Clear the search or choose another group.</div>}</div></section>;
}

export function DataExplorer({ evidence, series, research }: { evidence: Evidence; series: Series; research: Research }) {
  return <><SectionHeading number="02" title="The data behind the signal." description="Inspect the actual observations, see what is missing, and follow the definitions from raw fields to engineered features." /><Tabs.Root defaultValue="series"><Tabs.List className="subnav" aria-label="Data views"><Tabs.Trigger value="series">Time series</Tabs.Trigger><Tabs.Trigger value="fields">Raw fields & missingness</Tabs.Trigger><Tabs.Trigger value="features">Feature dictionary</Tabs.Trigger></Tabs.List><Tabs.Content value="series"><Timeline evidence={evidence} series={series} /></Tabs.Content><Tabs.Content value="fields"><RawFields evidence={evidence} /></Tabs.Content><Tabs.Content value="features"><Features research={research} /></Tabs.Content></Tabs.Root></>;
}
