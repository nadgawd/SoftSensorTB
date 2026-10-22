"""Live checks: operating region (T² / SPE), input health and lab accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.deployment import monitor


def correlated(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame({"x1": a, "x2": a + 0.05 * rng.normal(size=n), "x3": b, "x4": b + 0.05 * rng.normal(size=n)})


def test_region_limits_flag_far_points_and_broken_correlations():
    data = correlated()
    region = monitor.fit_region(data)
    assert region is not None and region.loadings.shape[1] == 2 and region.spe_limit is not None
    t2, spe, notes = monitor.region_findings(region, data.to_numpy())
    assert np.mean(np.array(t2) > region.t2_limit) == pytest.approx(0.01, abs=0.005)

    far = np.array([[6.0, 6.0, 0.0, 0.0]])  # along the training correlation, but far away
    broken = np.array([[1.0, -1.0, 0.0, 0.0]])  # close by, but x1 and x2 disagree
    _, _, far_notes = monitor.region_findings(region, far)
    _, _, broken_notes = monitor.region_findings(region, broken)
    assert any("T²" in n for n in far_notes[0]) and not any("SPE" in n for n in far_notes[0])
    assert any("SPE" in n for n in broken_notes[0])


def test_region_needs_enough_rows():
    assert monitor.fit_region(correlated(n=10)) is None


def test_input_findings_report_missing_out_of_range_and_stuck_sensors():
    train = pd.DataFrame({"t": np.linspace(0, 100, 200), "g": ["A", "B"] * 100})
    stats = monitor.input_stats(train, ["t", "g", "time"], "time")
    assert set(stats) == {"t", "g"}
    live = pd.DataFrame({
        "t": list(np.linspace(10, 20, 10)) + [50.0] * 25 + [np.nan, 150.0, -50.0, 103.0],
        "g": ["A"] * 35 + [None, "A", "C", "B"],
    })
    new = np.zeros(len(live), dtype=bool)
    new[-5:] = True
    missing, warnings = monitor.input_findings(live, stats, new)
    assert missing[1] == ["t", "g"]
    assert any("stuck" in w for w in warnings[0])  # 25 identical readings in a row
    assert any("above the training range" in w for w in warnings[2])
    assert any("below the training range" in w for w in warnings[3]) and any("never appeared" in w for w in warnings[3])
    assert warnings[4] == []  # 103 is within the 5% margin of the range


def test_accuracy_alarm_needs_enough_labs_and_a_clear_rise():
    assert monitor.accuracy([], 1.0)["n"] == 0
    assert monitor.accuracy([3.0] * 4, 1.0)["alarm"] is False
    worse = monitor.accuracy([3.0] * 5, 1.0)
    assert worse["alarm"] is True and worse["rmse"] == pytest.approx(3.0) and "retraining" in worse["message"]
    assert monitor.accuracy([0.5, -0.5] * 10, 1.0)["alarm"] is False


def test_bias_update_moves_toward_the_latest_error():
    assert monitor.update_bias(0.0, 0.3, 2.0) == pytest.approx(0.6)
    assert monitor.update_bias(0.6, 0.3, 2.0) == pytest.approx(0.6 * 0.7 + 0.6)
    assert monitor.update_bias(1.0, 0.0, 5.0) == 1.0
