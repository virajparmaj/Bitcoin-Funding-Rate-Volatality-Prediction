import { ArrowRight, ArrowUpRight } from 'lucide-react';
import type { Evidence, Series } from './types';
import { Chart } from './components/Chart';
import { BitcoinArtwork } from './components/BitcoinArtwork';
import { Badge, SourceLink, bp, fixed, number } from './components/ui';

export function Overview({ evidence, series, navigate }: { evidence: Evidence; series: Series; navigate: (tab: string) => void }) {
  const rf = evidence.rf.metrics.find(metric => /random|rf/i.test(metric.label))!;
  const current = evidence.rf.metrics.find(metric => /current|persistence/i.test(metric.label))!;
  const daily = series.raw_daily.map(row => ({ label: row.date, values: [bp(row.funding_rate.mean)], low: bp(row.funding_rate.min), high: bp(row.funding_rate.max), count: row.funding_rate.count, incomplete: row.row_count !== 24 || row.funding_rate.missing_count > 0 }));
  return <>
    <section className="overview-intro">
      <div className="hero-copy"><div className="eyebrow"><span className="accent-square" /> A GROUP RESEARCH PROJECT · WITH A LATER EVALUATION AUDIT</div><h1>Bitcoin funding-rate<br /><span>prediction & volatility.</span></h1><p className="intro-description">Direction classification, conditional variance modeling, and next-observation funding-rate regression.</p></div>
      <div className="hero-art-panel"><span className="art-label">BTCUSDT / BINANCE FUTURES</span><BitcoinArtwork priority /><div className="art-caption"><span>THE STUDY WINDOW</span><strong>2020—2024</strong><span>January 2020–October 2024 · arrival UTC</span></div></div>
      <div className="research-question"><span className="mono small">THE RESEARCH QUESTION</span><p>How well can models explain funding-rate direction, conditional variance, and the next observed rate?</p><button className="text-button" onClick={() => navigate('methods')}>Explore the methods <ArrowUpRight size={16} /></button></div>
    </section>
    <div className="dataset-strip">
      <div><span className="mono meta-label">OBSERVATIONS</span><b className="mono">{number(evidence.dataset.row_count)}</b><span>Nominally hourly ticker data</span></div>
      <div><span className="mono meta-label">COVERAGE · ARRIVAL UTC</span><b className="mono date-stat">Jan 2020 — Oct 2024</b><span>Binance Futures · BTCUSDT</span></div>
      <div><span className="mono meta-label">RAW FIELDS</span><b className="mono">{evidence.dataset.column_count.toString().padStart(2, '0')}</b><span>Market, funding, and time</span></div>
      <div><span className="mono meta-label">FUNDING EVENT IDS</span><b className="mono">{number(evidence.dataset.funding_event_count)}</b><span>Not verified settlement labels</span></div>
    </div>
    <div className="overview-grid">
      <section className="panel raw-preview">
        <div className="panel-heading"><div><span className="eyebrow">01 / THE OBSERVED SIGNAL</span><h2>Funding indications over time</h2></div><button className="icon-button" aria-label="Explore raw data and signals" onClick={() => navigate('data')}><ArrowUpRight size={20} /></button></div>
        <Chart points={daily} names={['Daily mean']} unit="bp" label="Daily funding-rate indications by valid arrival date UTC" compact />
        <div className="panel-bottom"><span>Daily aggregates · valid arrival UTC · rate × 10,000 = bp</span><SourceLink>Data provenance</SourceLink></div>
      </section>
      <section className="panel result-preview">
        <div className="eyebrow">02 / THE EVALUATION LESSON</div><h2>Strong fit.<br />A stronger baseline.</h2><p>On the same saved rows, current-rate persistence outperforms the Random Forest.</p>
        <div className="score-pair"><div><span>Saved Random Forest</span><strong className="mono">{fixed(rf.r2)}</strong><span className="mono small">R²</span></div><div><span>Current-rate persistence</span><strong className="mono accent">{fixed(current.r2)}</strong><span className="mono small">R²</span></div></div>
        <Badge kind="verified">Recomputed from saved predictions</Badge><p className="small muted">n = {number(evidence.rf.n)} · next-observation target<br />Retrospective comparison; no fresh training run.</p><button className="text-button" onClick={() => navigate('evidence')}>Inspect the comparison <ArrowRight size={16} /></button>
      </section>
    </div>
    <div className="overview-foot"><BitcoinArtwork className="transition-art" /><div><p>Technical work, with its limits in view.</p><span>Four analytical tasks. Separate targets. Traceable evidence.</span></div><button className="text-button" onClick={() => navigate('findings')}>Read the findings <ArrowRight size={15} /></button></div>
  </>;
}
