"""Descriptive influence of the largest realized baseline errors; never a new fit."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from v2.paper_tools.supplement import ROOT, score_frame, verify_parent, write_json
from v2.src.funding_labels import sha256
from v2.src.reporting import common_events


def influential_events(frame, fraction):
    """Select unique event IDs using baseline errors, identically for every model.

    Ties break by event time. This is an outcome-conditioned diagnostic, not
    an admissible trading filter or a replacement for the untrimmed result.
    """
    if not 0 < fraction < 1:
        raise ValueError("Invalid influence fraction")
    consistency=frame.groupby('event_id')[['target','indication']].nunique()
    if (consistency > 1).any().any():
        raise ValueError("Models disagree on paired target or baseline")
    unique=frame[['event_id','target','indication']].drop_duplicates('event_id').copy()
    unique['baseline_error_bp']=abs(unique.target-unique.indication)*1e4
    n=int(np.ceil(len(unique)*fraction))
    return unique.sort_values(['baseline_error_bp','event_id'],ascending=[False,True]).head(n)


def main():
    package=ROOT/'v2/paper_package/2026-10-01';parent=ROOT/'v2/results/settlement_study_v2_1'
    manifest=verify_parent(parent)
    config=json.loads((ROOT/'v2/configs/paper_supplement_v1.json').read_text())
    data=pd.read_csv(parent/'predictions.csv.gz',parse_dates=['event_id'],float_precision='round_trip')
    data=common_events(data,manifest['protocol']['horizons'])
    data=data[data.variant.eq('main') & data.model.isin(['ridge','rf'])]
    output=package/'run_influence'
    identity=dict(parent_predictions_sha256=sha256(parent/'predictions.csv.gz'),
                  generator_sha256=sha256(Path(__file__)),fractions=[.01,.05],
                  config=config,reason='Training q99 tail was empty; descriptive evaluation-error influence, specified after observing that empty stratum')
    if output.exists():
        r=json.loads((output/'receipt.json').read_text())
        if r['identity']!=identity or any(sha256(output/k)!=v for k,v in r['outputs'].items()):
            raise ValueError('Changed influence run; preserve and version outputs')
        print('Verified influence run');return
    rows,selections=[],[]
    for h,frame in data.groupby('horizon'):
        for fraction in identity['fractions']:
            selected=influential_events(frame,fraction)
            selections.append(selected.assign(horizon=h,fraction=fraction))
            for model,part in frame.groupby('model'):
                kept=part[~part.event_id.isin(selected.event_id)]
                rows.append(dict(horizon=h,model=model,removed_fraction=fraction,
                                 removed_n=len(selected),interpretation='descriptive_no_refit',
                                 **score_frame(kept,config)))
    output.mkdir()
    pd.DataFrame(rows).to_csv(output/'influence_scores.csv',index=False)
    pd.concat(selections).to_csv(output/'excluded_events.csv',index=False)
    write_json(output/'receipt.json',dict(identity=identity,outputs={p.name:sha256(p) for p in output.iterdir()}))
    print(pd.DataFrame(rows).query('horizon==4').to_string(index=False))


if __name__=='__main__':
    main()
