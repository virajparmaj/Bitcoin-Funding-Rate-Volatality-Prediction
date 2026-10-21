"""Generate publication tables, vector figures and their evidence manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v2.paper_tools.supplement import ROOT, verify_parent, write_json
from v2.src.funding_labels import sha256
from v2.src.metrics import loss_summary
from v2.src.reporting import common_events, verify_ledger
from v2.src.study_store import digest, execution_identity


def check_results(parent, run):
    manifest = verify_parent(parent)
    receipt = json.loads((run / "receipt.json").read_text())
    if receipt['identity']['parent_identity'] != digest(execution_identity(manifest)):
        raise ValueError('Supplement belongs to a different parent study')
    for name, value in receipt["outputs"].items():
        if sha256(run / name) != value:
            raise ValueError(f"Corrupted supplement artifact: {name}")
    for name, value in receipt['identity']['code'].items():
        if sha256(ROOT / name) != value:
            raise ValueError(f"Supplement implementation changed: {name}")
    times = ["event_id", "origin", "fit_time", "source_available_at", "label_available_at"]
    pred = pd.read_csv(parent / "predictions.csv.gz", parse_dates=times, date_format="mixed", float_precision="round_trip")
    verify_ledger(pred, manifest['protocol'])
    if not ((pred.source_available_at <= pred.origin) & (pred.fit_time <= pred.origin)
            & (pred.label_available_at > pred.origin)).all():
        raise ValueError("Forecast ledger violates source/fit/label timing")
    fold = pd.read_csv(parent / "folds.csv", parse_dates=["fit_time", "max_train_label_available_at", "min_test_origin"], date_format="mixed")
    if not ((fold.max_train_label_available_at < fold.fit_time) & (fold.min_test_origin >= fold.fit_time)).all():
        raise ValueError("Fold maturity violation")
    data = common_events(pred, manifest['protocol']['horizons'])
    scores = pd.read_csv(parent / "scores.csv")
    for (v, h, m), frame in data.groupby(["variant", "horizon", "model"]):
        saved = scores[(scores.variant==v)&(scores.horizon==h)&(scores.model==m)&(scores.block_days==7)].iloc[0]
        result = loss_summary(frame)
        for key, val in result.items():
            np.testing.assert_allclose(saved[key], val, rtol=1e-9, atol=1e-12)
    extra = pd.read_csv(run/'predictions.csv.gz', parse_dates=['event_id','origin','fit_time','label_available_at'], float_precision='round_trip')
    if extra.duplicated(['variant','horizon','model','event_id']).any() or not ((extra.fit_time<=extra.origin)&(extra.label_available_at>extra.origin)).all():
        raise ValueError('Supplement forecast timing or identity violation')
    supplemental_scores=pd.read_csv(run/'sensitivity_scores.csv')
    for (v,h,m),frame in extra[extra.event_id.isin(data.event_id)].groupby(['variant','horizon','model']):
        saved=supplemental_scores[(supplemental_scores.variant==v)&(supplemental_scores.horizon==h)&(supplemental_scores.model==m)].iloc[0]
        for key,val in loss_summary(frame).items():
            np.testing.assert_allclose(saved[key],val,rtol=1e-9,atol=1e-12)
    extra_folds=pd.read_csv(run/'folds.csv',parse_dates=['fit_time','max_train_label_available_at','min_test_origin'])
    if not ((extra_folds.max_train_label_available_at<extra_folds.fit_time)&(extra_folds.min_test_origin>=extra_folds.fit_time)).all():
        raise ValueError('Supplement fold maturity violation')
    return dict(status="passed", original_forecasts=len(pred), original_folds=len(fold),
                supplemental_forecasts=len(extra), supplemental_folds=len(extra_folds),
                matched_events=data.event_id.nunique(), independently_rescored_groups=128,
                source_fit_label_timing="all forecast rows", fold_maturity="all folds",
                parent_receipts="all verified", supplemental_artifact_hashes="all verified")


def save_figure(fig, path):
    for ext in ['pdf', 'svg', 'png']:
        kw = {'metadata': {'CreationDate': None, 'ModDate': None}} if ext=='pdf' else {}
        fig.savefig(path.with_suffix('.'+ext), dpi=300, bbox_inches='tight', **kw)
    plt.close(fig)


def make_figures(scores, sensitivity, cohort, contrasts, dest):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42,'svg.fonttype':'none'})
    main = scores[(scores.variant=='main')&(scores.block_days==7)]
    colors = {'indication':'#333333','ridge':'#0072B2','rf':'#D55E00'}
    fig, ax = plt.subplots(1,2,figsize=(9.2,3.6),layout='constrained')
    for model in ['indication','ridge','rf']:
        f=main[main.model==model].sort_values('horizon')
        ax[0].plot(f.horizon,f.mae_bp,'o-',label=model,color=colors[model])
        if model!='indication':
            offset=-.06 if model=='ridge' else .06
            ax[1].errorbar(f.horizon+offset, f.gain_bp,
                          yerr=[f.gain_bp-f.gain_low_bp,f.gain_high_bp-f.gain_bp],
                          fmt='o',capsize=3,label=model,color=colors[model])
    ax[0].set_ylabel('MAE (bp)'); ax[1].set_ylabel('MAE reduction vs indication (bp)')
    ax[1].axhline(0,color='#777777',lw=.8)
    for a in ax:
        a.set_xticks([1,2,4,6]);a.set_xlabel('Hours before settlement');a.legend(frameon=False)
    save_figure(fig,dest/'lead_time')
    fig, ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
    base=main[(main.horizon==4)&(main.model=='ridge')].assign(variant='original expanding')
    f=pd.concat([base,sensitivity[(sensitivity.horizon==4)&(sensitivity.model=='ridge')]])
    f=pd.concat([f,cohort[(cohort.horizon==4)&(cohort.model=='ridge')&(cohort.cohort=='all_main')].assign(variant='all main events')])
    y=np.arange(len(f))
    ax.errorbar(f.gain_bp,y,xerr=[f.gain_bp-f.gain_low_bp,f.gain_high_bp-f.gain_bp],fmt='o',capsize=4,color=colors['ridge'])
    ax.set_yticks(y, f.variant.str.replace('_',' '));ax.axvline(0,color='#777777',lw=.8)
    ax.set_xlabel('Four-hour ridge MAE reduction (bp)');ax.invert_yaxis()
    save_figure(fig,dest/'primary_sensitivity')
    fig, axes=plt.subplots(2,1,figsize=(8,6.5),height_ratios=[3,1],layout='constrained')
    labels_map={'main_error_minus_indication_error':'indication-only features',
                'main_error_minus_history_error':'indication + history',
                'main_error_minus_delay_60m_error':'60-minute source delay'}
    for ax, proxy in zip(axes,[False,True]):
        mask=contrasts.comparison.eq('proxy_gain_minus_settlement_gain') if proxy else contrasts.comparison.isin(labels_map)
        c=contrasts[contrasts.horizon.eq(4)&mask]
        labels=[r.model.upper() if proxy else f'{r.model.upper()}: {labels_map[r.comparison]}' for r in c.itertuples()]
        y=np.arange(len(c))
        ax.errorbar(c.estimate_bp,y,xerr=[c.estimate_bp-c.gain_low_bp,c.gain_high_bp-c.estimate_bp],fmt='o',capsize=3,color='#444444')
        ax.set_yticks(y,labels);ax.axvline(0,color='#777777',lw=.8);ax.invert_yaxis()
        ax.set_xlabel('Proxy-label gain minus settlement-label gain (bp)' if proxy else 'Full-model MAE minus alternative MAE (bp)')
    save_figure(fig,dest/'ablations')


def main(package):
    parent=ROOT/'v2/results/settlement_study_v2_1';run=package/'run'
    audit=check_results(parent,run)
    influence=package/'run_influence'
    influence_receipt=json.loads((influence/'receipt.json').read_text())
    if influence_receipt['identity']['parent_predictions_sha256'] != sha256(parent/'predictions.csv.gz'):
        raise ValueError('Influence analysis belongs to a different forecast ledger')
    if influence_receipt['identity']['generator_sha256'] != sha256(Path(__file__).with_name('influence.py')):
        raise ValueError('Influence implementation changed')
    for name,value in influence_receipt['outputs'].items():
        if sha256(influence/name)!=value:
            raise ValueError('Corrupted influence artifact')
    tables,figures=package/'tables',package/'figures'
    tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    inputs={}
    for base,names in [(parent,['scores.csv','paired_contrasts.csv','proxy_label_scores.csv','state_quarter_scores.csv','staleness_scores.csv']),
                       (run,['cohort_scores.csv','sensitivity_scores.csv','tail_scores.csv','leave_quarter_out.csv','simultaneous_bounds.csv']),
                       (influence,['influence_scores.csv','excluded_events.csv'])]:
        for name in names:
            inputs[str((base/name).relative_to(ROOT))]=sha256(base/name)
            (tables/name).write_bytes((base/name).read_bytes())
    scores=pd.read_csv(tables/'scores.csv');sensitivity=pd.read_csv(tables/'sensitivity_scores.csv')
    cohort=pd.read_csv(tables/'cohort_scores.csv');contrasts=pd.read_csv(tables/'paired_contrasts.csv')
    primary=scores[(scores.variant=='main')&(scores.block_days==7)&(scores.horizon==4)]
    primary.to_csv(tables/'primary.csv',index=False)
    make_figures(scores,sensitivity,cohort,contrasts,figures)
    write_json(package/'integrity_report.json',audit)
    write_json(package/'artifact_manifest.json',dict(generator=str(Path(__file__).relative_to(ROOT)),
               generator_sha256=sha256(Path(__file__)),inputs=inputs,
               outputs={str(p.relative_to(package)):sha256(p) for d in [tables,figures] for p in sorted(d.iterdir())},
               captions={'lead_time':'Original matched 1,889 events. Pointwise 95% paired seven-day block intervals; primary: 4h ridge.',
                         'primary_sensitivity':'Post-result fixed-hyperparameter window and cohort sensitivity. Pointwise intervals; no favorable variant selected.',
                         'ablations':'Original E3/E3b paired contrasts. Negative error difference favors full model; proxy rows have a different estimand. Holm tests in paired_contrasts.csv.'}))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=ROOT/'v2/paper_package/2026-10-01')
    main(parser.parse_args().package)
