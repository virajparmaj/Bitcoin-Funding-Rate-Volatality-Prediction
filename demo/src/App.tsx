import { Component, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import * as Tabs from '@radix-ui/react-tabs';
import { ArrowUpRight, Bitcoin, Clock3, Download, X } from 'lucide-react';
import type { Evidence, Research, Series } from './types';
import { SourceContext, number, fixed } from './components/ui';
import { Overview } from './Overview';
import { DataExplorer } from './DataExplorer';
import { Methods } from './Methods';
import { ModelEvidence } from './ModelEvidence';
import { Findings } from './Findings';

const NAV = [['overview', 'Overview'], ['data', 'Data & signals'], ['methods', 'Methods'], ['evidence', 'Models & evidence'], ['findings', 'Findings']];

class ErrorBoundary extends Component<{ children: ReactNode }, { error: boolean }> {
  state = { error: false };
  static getDerivedStateFromError() { return { error: true }; }
  render() {
    if (this.state.error) return <div className="load-state"><h1>The evidence view could not render.</h1><p>Reload to restore the local view. No research files have been changed.</p><button className="button" onClick={() => location.reload()}>Reload demo</button></div>;
    return this.props.children;
  }
}

async function readJson<T,>(file: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${file}`);
  if (!response.ok) throw new Error(`${file} could not be loaded (${response.status}).`);
  return response.json();
}

function Demo() {
  const [data, setData] = useState<{ evidence: Evidence; series: Series; research: Research } | null>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [tab, setTab] = useState(NAV.some(([key]) => key === location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
  const [summary, setSummary] = useState(false);
  const [sourceIds, setSourceIds] = useState<string[] | null>(null);
  const sourceReturnFocus = useRef<HTMLElement | null>(null);
  useEffect(() => {
    let active = true;
    setError('');
    Promise.all([readJson<Evidence>('evidence.json'), readJson<Series>('series.json'), readJson<Research>('research-content.json')]).then(([evidence, series, research]) => { if (active) setData({ evidence, series, research }); }).catch(reason => { if (active) setError(String(reason.message)); });
    return () => { active = false; };
  }, [attempt]);
  useEffect(() => {
    const onHash = () => {
      const hash = location.hash.slice(1);
      if (NAV.some(([key]) => key === hash)) setTab(hash);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const navigate = (value: string) => { setTab(value); history.replaceState(null, '', `#${value}`); window.scrollTo({ top: 0, behavior: 'instant' }); };
  if (!data) return <div className="load-state"><div className="brand">f<span>/</span> funding research</div><h1>{error ? 'The evidence files are unavailable.' : 'Opening the research notebook.'}</h1><p>{error || 'Reading deterministic local exports…'}</p>{error && <><p>Run the export instructions in demo/README.md, then retry.</p><button className="button" onClick={() => setAttempt(attempt + 1)}>Retry loading evidence</button></>}</div>;
  const { evidence, series, research } = data;
  const allSources = [...evidence.sources, ...research.sources];
  const selectedSources = sourceIds?.length ? allSources.filter(source => sourceIds.includes(source.id)) : allSources;
  const rf = evidence.rf.metrics.find(metric => /random|rf/i.test(metric.label))!;
  const persistence = evidence.rf.metrics.find(metric => /current|persistence/i.test(metric.label))!;
  return <SourceContext.Provider value={ids => { sourceReturnFocus.current = document.activeElement as HTMLElement | null; setSourceIds(ids ?? []); }}>
    <a className="skip-link" href="#main" onClick={event => { event.preventDefault(); document.getElementById('main')?.focus(); }}>Skip to research content</a>
    <div className="app-shell"><header className="app-header"><a className="brand" href="#overview" onClick={() => navigate('overview')}><span className="brand-glyph" aria-hidden="true"><Bitcoin size={24} strokeWidth={2} /></span><span>funding<span className="brand-secondary"> / research</span></span></a><div className="header-meta mono">BTCUSDT <span>·</span> RESEARCH STUDY</div><button className="button summary-trigger" onClick={() => setSummary(true)}><Clock3 size={15} />View 60-second summary<ArrowUpRight size={14} /></button></header>
      <Tabs.Root value={tab} onValueChange={navigate}><div className="navigation-row"><Tabs.List className="navigation" aria-label="Research sections">{NAV.map(([id, title], index) => <Tabs.Trigger key={id} value={id}><span className="mono">0{index + 1}</span>{title}</Tabs.Trigger>)}</Tabs.List><span className="local-indicator mono"><span />LOCAL EVIDENCE</span></div>
        <main id="main" tabIndex={-1}>
          <Tabs.Content value="overview"><Overview evidence={evidence} series={series} navigate={navigate} /></Tabs.Content>
          <Tabs.Content value="data"><DataExplorer evidence={evidence} series={series} research={research} /></Tabs.Content>
          <Tabs.Content value="methods"><Methods research={research} /></Tabs.Content>
          <Tabs.Content value="evidence"><ModelEvidence evidence={evidence} series={series} research={research} /></Tabs.Content>
          <Tabs.Content value="findings"><Findings evidence={evidence} research={research} /></Tabs.Content>
        </main>
      </Tabs.Root><footer className="app-footer"><span>STAT 429 GROUP PROJECT <span className="footer-divider">/</span> Original research + later artifact verification</span><a href={`${import.meta.env.BASE_URL}data/manifest.json`} download><Download size={12} />Evidence manifest</a><span className="mono">NO LIVE INFERENCE</span></footer></div>
    <Dialog.Root open={summary} onOpenChange={setSummary}>
      <Dialog.Portal><Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog-content summary-dialog" onCloseAutoFocus={event => { event.preventDefault(); document.querySelector<HTMLButtonElement>('.summary-trigger')?.focus(); }}>
          <Dialog.Close className="dialog-close" aria-label="Close summary"><X size={20} /></Dialog.Close>
          <div className="eyebrow">THE STUDY / IN 60 SECONDS</div>
          <Dialog.Title>Good models need<br /><span className="muted">good comparisons.</span></Dialog.Title>
          <Dialog.Description className="sr-only">The problem, data, work, representative result, and learning from the project.</Dialog.Description>
          <dl className="summary-list">
            <div><dt>01 / Problem</dt><dd>Investigate funding-rate direction, conditional variability, and next-observation values.</dd></div>
            <div><dt>02 / Data</dt><dd>{number(evidence.dataset.row_count)} nominally hourly Binance BTCUSDT observations, January 2020–October 2024.</dd></div>
            <div><dt>03 / Work</dt><dd>Exploratory statistics and feature engineering; classification, GARCH-family fitting, and regression; later saved-result verification.</dd></div>
            <div className="summary-result"><dt>04 / Result</dt><dd><span className="mono">RF R² {fixed(rf.r2)}</span><span className="muted"> vs </span><span className="mono accent">persistence {fixed(persistence.r2)}</span><small>On the same {number(evidence.rf.n)} saved rows. Retrospective diagnostic, no fresh training.</small></dd></div>
            <div><dt>05 / Learning</dt><dd>Model quality depends on target timing, appropriate baselines, and reproducible evaluation.</dd></div>
          </dl>
          <button className="button" onClick={() => { setSummary(false); navigate('evidence'); }}>Explore the evidence <ArrowUpRight size={15} /></button>
          <p className="small muted">Group coursework and later audit work are separate parts of this research record.</p>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
    <Dialog.Root open={sourceIds !== null} onOpenChange={open => { if (!open) setSourceIds(null); }}><Dialog.Portal><Dialog.Overlay className="dialog-overlay" /><Dialog.Content className="dialog-content source-dialog" onCloseAutoFocus={event => { event.preventDefault(); sourceReturnFocus.current?.focus(); }}><Dialog.Close className="dialog-close" aria-label="Close sources"><X size={20} /></Dialog.Close><div className="eyebrow">FOLLOW THE EVIDENCE</div><Dialog.Title>Source record</Dialog.Title><Dialog.Description>Local excerpts and file fingerprints from the inspected repository. A stored result is not a fresh model run.</Dialog.Description><div className="source-list">{selectedSources?.map(source => <article className="source-record" key={source.id}><h3 className="mono">{source.path}</h3><p>{source.locator}</p><pre>{source.excerpt}</pre>{source.sha256 && <details><summary>SHA-256 fingerprint</summary><code>{source.sha256}</code></details>}</article>)}</div></Dialog.Content></Dialog.Portal></Dialog.Root>
  </SourceContext.Provider>;
}

export function App() { return <ErrorBoundary><Demo /></ErrorBoundary>; }
