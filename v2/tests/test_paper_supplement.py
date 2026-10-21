"""Scientific failure cases for the additive paper sensitivities."""
import numpy as np
import pandas as pd
import pytest

from v2.paper_tools.supplement import calibration_prediction, restrict_training, simultaneous_bounds


def test_calibration_uses_only_training_residuals():
    train = pd.DataFrame({"residual": [0., .0001, .0002]})
    test = pd.DataFrame({"indication": [0., .0001], "target": [99., -99.]})
    for model in ("residual_mean", "residual_median"):
        np.testing.assert_allclose(calibration_prediction(train, test, model), [.0001, .0002])
        modified = test.assign(target=0)
        np.testing.assert_array_equal(calibration_prediction(train, test, model),
                                      calibration_prediction(train, modified, model))


def test_calendar_window_boundary_and_label_maturity():
    cutoff = pd.Timestamp("2024-03-01", tz="UTC")
    events = pd.to_datetime(["2023-02-28", "2023-03-01", "2024-02-29"], utc=True)
    train = pd.DataFrame({"event_id": events, "label_available_at": events+pd.Timedelta(minutes=5)})
    assert restrict_training(train, cutoff, 12).index.tolist() == [1, 2]
    with pytest.raises(ValueError, match="Unmatured"):
        restrict_training(train.assign(label_available_at=cutoff), cutoff, 12)


def family():
    events = pd.date_range("2023-01-01", periods=60, freq="2D", tz="UTC")
    return pd.DataFrame([dict(event_id=e, horizon=h, model=m, target=.0001,
                              indication=0., prediction=.00005)
                         for e in events for h in [1,2,4,6] for m in ['ridge','rf']])


def settings():
    return dict(block_days=7, bootstrap_draws=200, bootstrap_seed=42)


def test_joint_bounds_keep_paired_constants_and_gaps():
    result = simultaneous_bounds(family(), settings())
    assert len(result)==8 and result.family_size.eq(8).all()
    np.testing.assert_allclose(result.gain_bp, .5)
    np.testing.assert_allclose(result.simultaneous_low_bp, .5)
    pd.testing.assert_frame_equal(result, simultaneous_bounds(family(), settings()))


def test_joint_bounds_reject_missing_or_duplicate_pairs():
    data=family()
    with pytest.raises(ValueError, match="matched"):
        simultaneous_bounds(data.iloc[1:], settings())
    with pytest.raises(ValueError, match="Duplicate"):
        simultaneous_bounds(pd.concat([data,data.iloc[:1]]), settings())


def test_joint_bounds_enclose_estimates_for_variable_paired_gains():
    data=family()
    data['prediction']=np.random.default_rng(11).normal(.00001,.00003,len(data))
    result=simultaneous_bounds(data, settings())
    assert (result.simultaneous_low_bp < result.gain_bp).all()
    assert (result.simultaneous_high_bp > result.gain_bp).all()


def test_joint_bounds_reject_short_calendar():
    with pytest.raises(ValueError, match="Insufficient"):
        simultaneous_bounds(family().iloc[:16], settings())


def test_influence_uses_one_paired_mask_without_modifying_input():
    from v2.paper_tools.publication.influence import influential_events
    data=family().query('horizon==4').copy()
    data.loc[data.event_id.eq(data.event_id.max()),'target']=.1
    original=data.copy(deep=True)
    selected=influential_events(data,.01)
    assert selected.event_id.tolist()==[data.event_id.max()]
    pd.testing.assert_frame_equal(data,original)
    bad=data.copy();bad.loc[bad.index[0],'indication']=1
    with pytest.raises(ValueError,match='disagree'):
        influential_events(bad,.01)
