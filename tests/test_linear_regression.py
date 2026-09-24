"""Linear regression is offered for training (as OLS) and as an RFE estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LinearRegression

from backend.mcp_server.eda_tools import SelectFeaturesRFEArgs
from backend.mcp_server.modeling_tools import (
    SoftSensorAlgorithm,
    TrainSoftSensorArgs,
    _fit_estimator,
)

_VID = "00000000-0000-4000-8000-000000000000"


def _train_args(algorithm: str) -> TrainSoftSensorArgs:
    return TrainSoftSensorArgs(
        dataset_version_id=_VID,
        algorithm=algorithm,
        target_column="y",
        feature_columns=["a", "b"],
    )


@pytest.mark.parametrize(
    "name", ["OLS", "ols", "linear", "Linear Regression", "linear_regression", "LR", "MLR"]
)
def test_linear_regression_names_map_to_ols(name):
    assert _train_args(name).algorithm is SoftSensorAlgorithm.OLS


def test_other_algorithms_still_parse_and_unknown_is_rejected():
    assert _train_args("ridge").algorithm is SoftSensorAlgorithm.RIDGE
    assert _train_args("k-nn").algorithm is SoftSensorAlgorithm.KNN
    with pytest.raises(ValidationError):
        _train_args("random_forest")


def test_ols_fits_plain_linear_regression():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
    y = 3 * X["a"] - 2 * X["b"] + 1
    model = _fit_estimator(SoftSensorAlgorithm.OLS, X, y, n_components=2)
    assert type(model) is LinearRegression
    assert model.coef_ == pytest.approx([3, -2])
    assert model.intercept_ == pytest.approx(1)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("linear", "linear"), ("OLS", "linear"), ("Linear Regression", "linear"),
     ("ridge", "ridge"), ("Lasso", "lasso")],
)
def test_rfe_accepts_linear_estimator(given, expected):
    args = SelectFeaturesRFEArgs(dataset_version_id=_VID, target_column="y", estimator=given)
    assert args.estimator == expected


def test_rfe_rejects_unknown_estimator():
    with pytest.raises(ValidationError):
        SelectFeaturesRFEArgs(dataset_version_id=_VID, target_column="y", estimator="svm")
