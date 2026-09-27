import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';
import { ArrowUpRight, FileText } from 'lucide-react';
import { BitcoinArtwork } from './BitcoinArtwork';

export const SourceContext = createContext<(ids?: string[]) => void>(() => undefined);

export function SourceLink({ ids, children = 'View source' }: { ids?: string[]; children?: ReactNode }) {
  const open = useContext(SourceContext);
  return <button className="source-link" onClick={() => open(ids)}><FileText size={12} />{children}<ArrowUpRight size={12} /></button>;
}

export function Badge({ children, kind = 'neutral' }: { children: ReactNode; kind?: 'verified' | 'neutral' | 'planned' }) {
  return <span className={`badge badge-${kind}`}><span />{children}</span>;
}

export function SectionHeading({ number, title, description }: { number: string; title: string; description: string }) {
  return <div className={`section-heading section-heading-${number}`}><div className="section-heading-copy"><div className="eyebrow">BTCUSDT / RESEARCH NOTE {number}</div><h1>{title}</h1><p>{description}</p></div><BitcoinArtwork className="section-art" priority /></div>;
}

export function Note({ children }: { children: ReactNode }) {
  return <div className="note"><span className="note-marker">i</span><div>{children}</div></div>;
}

export function MetricTable({ children, caption }: { children: ReactNode; caption: string }) {
  return <div className="table-scroll"><table className="metric-table"><caption>{caption}</caption>{children}</table></div>;
}

export const number = (value: number) => value.toLocaleString('en-US');
export const fixed = (value: number, digits = 5) => value.toFixed(digits);
export const bp = (value: number | null) => value === null ? null : value * 10_000;
