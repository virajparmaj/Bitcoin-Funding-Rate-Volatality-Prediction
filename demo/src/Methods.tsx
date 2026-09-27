import * as Tabs from '@radix-ui/react-tabs';
import type { Research, Task } from './types';
import { Badge, Note, SectionHeading, SourceLink } from './components/ui';

function TaskDetail({ task }: { task: Task }) {
  return <section className="panel task-detail">
    <div className="panel-heading"><div><div className="eyebrow">{task.subtitle}</div><h2>{task.title}</h2></div><Badge kind={task.status.startsWith('Recomputed') ? 'verified' : 'neutral'}>{task.status}</Badge></div>
    <div className="target-definition"><span className="eyebrow">WHAT IS PREDICTED OR FITTED</span><code>{task.target}</code><p>{task.units}</p></div>
    <div className="method-columns"><div><h3>Inputs</h3><ul className="plain-list">{task.inputs.map(input => <li key={input}>{input}</li>)}</ul><h3>Models in the experiment</h3><ul className="plain-list">{task.models.map(model => <li key={model}>{model}</li>)}</ul></div><div><h3>Evaluation</h3><p>{task.split}</p><h3>Preprocessing</h3><ul className="plain-list">{task.preprocessing.map(step => <li key={step}>{step}</li>)}</ul></div></div>
    <details className="disclosure" open><summary>Interpretation & limitations</summary><ul className="plain-list">{task.limitations.map(limitation => <li key={limitation}>{limitation}</li>)}</ul></details>
    <SourceLink ids={task.sourceIds}>Inspect this task’s implementation</SourceLink>
  </section>;
}

export function Methods({ research }: { research: Research }) {
  return <><SectionHeading number="03" title="Four tasks. Different questions." description="Follow each experiment from its inputs to its target and evaluation. These tasks share a dataset, but their results measure different things." />
    <div className="method-origin"><span className="mono">BINANCE BTCUSDT TICKER OBSERVATIONS</span><span>Shared funding history, market state, and engineered features</span></div>
    <Tabs.Root defaultValue="analysis-a"><Tabs.List className="task-navigation" aria-label="Analytical tasks">{research.tasks.map((task, index) => <Tabs.Trigger key={task.id} value={task.id}><span className="mono task-number">{index === 0 ? 'A' : `M${index}`}</span><span><b>{task.title}</b><small>{task.subtitle}</small></span></Tabs.Trigger>)}</Tabs.List>{research.tasks.map(task => <Tabs.Content key={task.id} value={task.id}><TaskDetail task={task} /></Tabs.Content>)}</Tabs.Root>
    <div className="method-connections"><Note><strong>Actual connection:</strong> GARCH fitting consumes the funding-rate series. There is no implemented direction-classifier → GARCH dependency in the inspected fitting cell.</Note><Note><strong>Intended integration:</strong> Model 3 attempts to add direction and conditional-variance outputs. Both are constant in the saved RF test data; a benefit from the integration is unverified.</Note></div>
    <SourceLink ids={['garch', 'regression-rf', 'saved-integration-values']}>Connection evidence</SourceLink>
  </>;
}
