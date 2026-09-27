import { useMemo, useState } from 'react';
import type { Evidence, Research, Series } from './types';
import { Chart } from './components/Chart';
import { Badge, MetricTable, Note, SectionHeading, SourceLink, fixed, number } from './components/ui';

const TASK_OPTIONS = [
  ['rf', 'Model 3 · Next-observation regression'],
  ['analysis', 'Analysis A · Contemporaneous regression'],
  ['direction', 'Model 1 · Direction classification'],
  ['variance', 'Model 2 · Conditional variance'],
  ['sarimax', 'Model 3 · SARIMAX diagnostic'],
];

function PredictionExplorer({ series, n }: { series: Series; n: number }) {
  const [view, setView] = useState('predictions');
  const [window, setWindow] = useState('all');
  const largestErrorIndex = useMemo(() => series.rf.reduce((best, row, index, rows) => Math.abs(row.residual) > Math.abs(rows[best].residual) ? index : best, 0), [series]);
  let rows = series.rf;
  if (window === 'first') rows = rows.slice(0, 200);
  if (window === 'last') rows = rows.slice(-200);
  if (window === 'error') rows = rows.slice(Math.max(0, largestErrorIndex - 100), Math.min(rows.length, largestErrorIndex + 101));
  const residual = view === 'residuals';
  const points = rows.map(row => ({
    label: `Row ${number(row.row_index)}`, x: row.row_index,
    values: residual ? [row.residual * 10_000, (row.actual - row.current) * 10_000] : [row.actual * 10_000, row.predicted * 10_000, row.current * 10_000],
    detail: 'One saved pair · no aggregation',
  }));
  return <section className="panel prediction-panel"><div className="panel-heading"><div><div className="eyebrow">INSPECT THE SAVED PAIRS</div><h2>{residual ? 'Where the errors occur' : 'Actual, predicted, and persistent'}</h2></div><SourceLink ids={['rf', 'rf-target']}>Saved predictions</SourceLink></div>
    <div className="filter-bar"><label>Chart view<select value={view} onChange={event => setView(event.target.value)}><option value="predictions">Actual versus predicted</option><option value="residuals">Residuals · actual minus predictor</option></select></label><label>Saved-row window<select value={window} onChange={event => setWindow(event.target.value)}><option value="all">All saved rows</option><option value="first">First 200 saved rows</option><option value="last">Last 200 saved rows</option><option value="error">Around the largest RF error</option></select></label><span className="window-count mono">{number(rows.length)} / {number(n)} pairs displayed</span></div>
    <Chart key={`${window}-${view}`} points={points} names={residual ? ['RF residual', 'Persistence residual'] : ['Actual next rate', 'Saved RF', 'Current-rate persistence']} unit="bp" label={residual ? 'Saved RF and persistence residuals by saved row index' : 'Actual versus predicted funding rates by saved row index'} />
    <p className="chart-caption">All selected pairs are drawn, without downsampling. The x-axis is the zero-based saved-row index, not UTC or a fixed forecast lead time. Rate × 10,000 = basis points. Residual = actual next rate − predictor. Window selection changes only this chart; the original {number(n)}-row metrics above remain fixed.</p>
  </section>;
}

function ErrorComparison({ evidence }: { evidence: Evidence }) {
  const max = Math.max(...evidence.rf.metrics.map(metric => metric.mae_bp));
  return <div className="error-comparison"><div className="eyebrow">MEAN ABSOLUTE ERROR / LOWER IS BETTER</div>{evidence.rf.metrics.map(metric => <div className="error-bar-row" key={metric.id}><span>{metric.label}</span><div className="error-bar-track"><div className={metric.id === 'current' ? 'best-bar' : ''} style={{ width: `${metric.mae_bp / max * 100}%` }} /></div><b className="mono">{fixed(metric.mae_bp, 3)} bp</b></div>)}<p className="small muted">Identical {number(evidence.rf.n)} saved pairs · full-sample MAE</p></div>;
}

function RFResults({ evidence, series }: { evidence: Evidence; series: Series }) {
  return <><section className="panel"><div className="panel-heading"><div><div className="eyebrow">MODEL 3 / NEXT OBSERVATION</div><h2>A matched comparison with persistence</h2></div><Badge kind="verified">Recomputed from saved predictions</Badge></div><p className="panel-description">Target: <code>funding_rate[t+1]</code>. Every predictor below uses the same {number(evidence.rf.n)} saved RF test rows. Baselines were added later as artifact diagnostics.</p>
    <MetricTable caption="R² is unitless, not percentage accuracy. Errors use basis points (native rate × 10,000). These are saved-artifact scores; no training or new holdout was run."><thead><tr><th>Predictor</th><th className="numeric">R²</th><th className="numeric">MAE · bp</th><th className="numeric">RMSE · bp</th><th className="numeric">n</th></tr></thead><tbody>{evidence.rf.metrics.map(metric => <tr className={metric.id === 'current' ? 'best' : ''} key={metric.id}><td>{metric.label}{metric.id === 'current' && <span className="table-tag">RECONSTRUCTED CURRENT OBSERVATION</span>}{metric.id === 'lag1' && <span className="table-tag">ONE OBSERVATION OLDER THAN CURRENT</span>}</td><td className="numeric">{fixed(metric.r2, 6)}</td><td className="numeric">{fixed(metric.mae_bp, 6)}</td><td className="numeric">{fixed(metric.rmse_bp, 6)}</td><td className="numeric">{number(metric.n)}</td></tr>)}</tbody></MetricTable>
    <div className="comparison-detail"><ErrorComparison evidence={evidence} /><div><div className="excess-error"><strong className="mono">+{fixed(evidence.rf.comparison.excess_mse_pct, 2)}%</strong><span>RF MSE versus current-rate persistence</span></div><div className="excess-error"><strong className="mono">+{fixed(evidence.rf.comparison.excess_mae_pct, 2)}%</strong><span>RF MAE versus current-rate persistence</span></div><SourceLink ids={['rf', 'rf-target', 'research-scores']}>Scores & evaluation source</SourceLink></div></div>
    <details className="disclosure"><summary>How the persistence baseline is reconstructed</summary><code className="formula-block">current_rate[t] = 3 × funding_rate_ma3[t] − funding_rate_lag1[t] − funding_rate_lag2[t]</code><p>The rolling mean contains the current rate and the two previous observations. This identity recovers the current observation; Model 3’s target is the next observation. It does not establish future-target leakage. The later diagnostic compares this current-rate predictor with the saved RF predictions, without claiming a clean rerun.</p><SourceLink ids={['technical-definitions', 'lag-definitions', 'rf-target', 'research-evidence']}>Formula and timing evidence</SourceLink></details>
    <details className="disclosure"><summary>Inspect constant integration outputs</summary><div className="table-scroll"><table><thead><tr><th>Saved RF input</th><th>Distinct test values</th><th>Value · native units</th></tr></thead><tbody>{['model1_direction_pred', 'model2_volatility_h1'].map(name => { const record = evidence.rf.constant_columns[name]; return <tr key={name}><td><code>{name}</code></td><td className="mono">{record.distinct_count}</td><td className="mono">{record.values.join(', ')}</td></tr>; })}</tbody></table></div><p>These test inputs are constant. Their benefit is not demonstrated, and test constants alone do not establish their effects during training.</p><SourceLink ids={['saved-integration-values', 'regression-rf']}>Integration evidence</SourceLink></details>
    </section><PredictionExplorer series={series} n={evidence.rf.n} /></>;
}

function AnalysisResults({ research }: { research: Research }) {
  return <section className="panel"><div className="panel-heading"><div><div className="eyebrow">ANALYSIS A / CURRENT OBSERVATION</div><h2>Exploratory contemporaneous regression</h2></div><Badge>Stored execution output</Badge></div><Note>The target is the current funding rate. Evaluation uses a shuffled 80/20 split. LR2 and Random Forest include a 24-row funding-rate standard deviation containing the current response.</Note>
    <MetricTable caption="Historical notebook outputs, not independently reproduced. The test sample count is not printed in the metric outputs; it is left unavailable rather than inferred as a verified count."><thead><tr><th>Model</th><th className="numeric">R²</th><th className="numeric">MSE · native rate²</th><th>Test n</th><th>Evidence</th></tr></thead><tbody>{research.metrics.filter(metric => metric.taskId === 'analysis-a').map(metric => <tr key={metric.id}><td>{metric.model}</td><td className="numeric">{fixed(metric.values.r2, 6)}</td><td className="numeric">{metric.values.mse.toExponential(6)}</td><td>Unavailable</td><td><SourceLink ids={metric.sourceIds}>Stored output</SourceLink></td></tr>)}</tbody></MetricTable>
    <div className="data-notes two-columns"><div><h3>What the scores establish</h3><p>They describe historical fits of contemporaneous funding rates. The printed “next five” regression values reuse test inputs and do not demonstrate a live future forecast.</p></div><div><h3>Additional statistical work</h3><p>The notebook includes time series, distributions, correlations, rolling statistics, ADF, ACF/PACF, and a full-series ARIMA(5,1,0) fit.</p><SourceLink ids={['analysis-adf-output', 'analysis-arima-code', 'analysis-arima-output']}>Statistical outputs</SourceLink></div></div>
  </section>;
}

function UnverifiedResults({ research, taskId }: { research: Research; taskId: string }) {
  const task = research.tasks.find(item => item.id === taskId)!;
  const history = research.historical.find(item => item.taskId === taskId);
  return <section className="panel"><div className="panel-heading"><div><div className="eyebrow">{task.subtitle}</div><h2>{task.title}</h2></div><Badge>Implemented, result unavailable</Badge></div><div className="target-definition"><span className="eyebrow">TARGET</span><code>{task.target}</code><p>{task.units}</p></div>
    <div className="model-cards">{task.models.map(model => <article className="model-card" key={model}><h3>{model}</h3><span className="mono small muted">Verified evaluation score: unavailable</span></article>)}</div>
    <div className="method-columns"><div><h3>Evaluation in source</h3><p>{task.split}</p></div><div><h3>What remains unresolved</h3><ul className="plain-list">{task.limitations.map(limitation => <li key={limitation}>{limitation}</li>)}</ul></div></div>
    <SourceLink ids={task.sourceIds}>Implementation & result status</SourceLink>
    {history && <details className="disclosure historical"><summary>Historical narrative · unverified result claims</summary><Badge>{history.status}</Badge><h3>{history.title}</h3><p>{history.text}</p><p className="small">{history.evaluation} Sample count: unavailable.</p><SourceLink ids={history.sourceIds}>Compare reported claims</SourceLink></details>}
  </section>;
}

function SarimaxResults({ evidence }: { evidence: Evidence }) {
  return <section className="panel"><div className="panel-heading"><div><div className="eyebrow">MODEL 3 / SEPARATE DIAGNOSTIC</div><h2>SARIMAX saved block predictions</h2></div><Badge kind="verified">Recomputed from saved predictions</Badge></div><div className="diagnostic-score"><div><span className="mono meta-label">SAVED-PAIR R²</span><strong className="mono">{fixed(evidence.sarimax.r2, 6)}</strong></div><div><span className="mono meta-label">SAVED PAIRS</span><strong className="mono">{number(evidence.sarimax.n)}</strong></div></div><Note>The sample, scaling, and block forecast setup differ from the RF comparison. This is not an identical rolling one-step evaluation, so it has its own diagnostic view.</Note><div className="method-columns"><div><h3>Target and forecast setup</h3><p>The notebook targets <code>future_funding_rate = funding_rate.shift(-1)</code>. It predicts a holdout block using test-period exogenous features, without demonstrated sequential target-state updating.</p></div><div><h3>Units and interpretation</h3><p>{evidence.sarimax.scale_note}</p><p>R² is unitless. A negative value indicates greater squared error than the evaluation-set mean benchmark; it is not a percentage accuracy or a trading-profit measure.</p></div></div><SourceLink ids={['sarimax', 'sarimax-setup']}>Saved file & forecast setup</SourceLink></section>;
}

export function ModelEvidence({ evidence, series, research }: { evidence: Evidence; series: Series; research: Research }) {
  const [task, setTask] = useState('rf');
  return <><SectionHeading number="04" title="Results, with their evidence attached." description="Keep targets and evaluation protocols separate. Headline metrics come from saved prediction pairs or stored notebook outputs." /><div className="evidence-task-select filter-bar"><label>Analytical task<select value={task} onChange={event => setTask(event.target.value)}>{TASK_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><p className="small muted">Changing tasks does not train or run a model.</p></div>
    {task === 'rf' && <RFResults evidence={evidence} series={series} />}{task === 'analysis' && <AnalysisResults research={research} />}{task === 'direction' && <UnverifiedResults research={research} taskId="model-1" />}{task === 'variance' && <UnverifiedResults research={research} taskId="model-2" />}{task === 'sarimax' && <SarimaxResults evidence={evidence} />}
    <details className="disclosure evidence-key"><summary>How to read the evidence labels</summary><dl className="feature-facts"><div><dt>Recomputed from saved predictions</dt><dd>Arithmetic reproduced from existing saved files; no fresh model training.</dd></div><div><dt>Stored execution output</dt><dd>Present in notebook outputs, not independently reproduced.</dd></div><div><dt>Reported in project notes</dt><dd>Narrative claim only, including notebook markdown. Discrepancies remain visible.</dd></div><div><dt>Implemented, result unavailable</dt><dd>Code exists but a usable result cannot be verified.</dd></div><div><dt>Planned</dt><dd>Future work, excluded from completed-work comparisons.</dd></div></dl></details>
  </>;
}
