"""
Unit tests for optimizers.
"""
import numpy as np
import pytest
from numpy.linalg import norm
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted

from pysindy.optimizers import ConstrainedSR3
from pysindy.optimizers import SINDyOptimizer
from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ
from pysindy.utils import supports_multiple_targets


class DummyLinearModel(BaseEstimator):
    # Does not natively support multiple targets
    def fit(self, x, y):
        self.coef_ = np.ones(x.shape[1])
        self.intercept_ = 0
        return self

    def predict(self, x):
        return x


class DummyEmptyModel(BaseEstimator):
    # Does not have fit or predict methods
    def __init__(self):
        self.fit_intercept = False
        self.normalize = False


class DummyModelNoCoef(BaseEstimator):
    # Does not set the coef_ attribute
    def fit(self, x, y):
        self.intercept_ = 0
        return self

    def predict(self, x):
        return x


@pytest.mark.parametrize(
    "cls, support",
    [
        (Lasso, True),
        (STLSQ, True),
        (SR3, True),
        (ConstrainedSR3, True),
        (DummyLinearModel, False),
    ],
)
def test_supports_multiple_targets(cls, support):
    assert supports_multiple_targets(cls()) == support


@pytest.fixture(params=["data_derivative_1d", "data_derivative_2d"])
def data(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ(),
        SR3(),
        ConstrainedSR3(),
        Lasso(fit_intercept=False),
        ElasticNet(fit_intercept=False),
        DummyLinearModel(),
    ],
)
def test_fit(data, optimizer):
    x, x_dot = data
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    opt = SINDyOptimizer(optimizer, unbias=False)
    opt.fit(x, x_dot)

    check_is_fitted(opt)
    assert opt.complexity >= 0
    if len(x_dot.shape) > 1:
        assert opt.coef_.shape == (x.shape[1], x_dot.shape[1])
    else:
        assert opt.coef_.shape == (1, x.shape[1])


@pytest.mark.parametrize(
    "kwargs", [{"normalize": True}, {"fit_intercept": True}, {"copy_X": False}]
)
def test_alternate_parameters(data_derivative_1d, kwargs):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    model = STLSQ(**kwargs)
    model.fit(x, x_dot)
    model.fit(x, x_dot, sample_weight=x[:, 0])

    check_is_fitted(model)


def test_bad_parameters():

    with pytest.raises(ValueError):
        STLSQ(threshold=-1)

    with pytest.raises(ValueError):
        STLSQ(alpha=-1)

    with pytest.raises(ValueError):
        STLSQ(max_iter=0)

    with pytest.raises(ValueError):
        SR3(threshold=-1)

    with pytest.raises(ValueError):
        SR3(nu=0)

    with pytest.raises(ValueError):
        SR3(tol=0)

    with pytest.raises(ValueError):
        SR3(max_iter=0)

    with pytest.raises(NotImplementedError):
        SR3(thresholder="l2")

    with pytest.raises(ValueError):
        ConstrainedSR3(threshold=-1)

    with pytest.raises(ValueError):
        ConstrainedSR3(nu=0)

    with pytest.raises(ValueError):
        ConstrainedSR3(tol=0)

    with pytest.raises(ValueError):
        ConstrainedSR3(max_iter=0)

    with pytest.raises(NotImplementedError):
        ConstrainedSR3(thresholder="l2")

    with pytest.raises(ValueError):
        ConstrainedSR3(thresholder="weighted_l0", thresholds=None)

    with pytest.raises(ValueError):
        ConstrainedSR3(thresholder="l0", thresholds=np.ones((5, 5)))

    with pytest.raises(ValueError):
        thresholds = np.ones((5, 5))
        thresholds[0, 0] = -1
        ConstrainedSR3(thresholds=thresholds)


def test_bad_optimizers(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    with pytest.raises(AttributeError):
        opt = SINDyOptimizer(DummyEmptyModel())

    with pytest.raises(AttributeError):
        opt = SINDyOptimizer(DummyModelNoCoef())
        opt.fit(x, x_dot)


# The different capitalizations are intentional;
# I want to make sure different versions are recognized
@pytest.mark.parametrize("thresholder", ["L0", "l1", "CAD"])
def test_sr3_prox_functions(data_derivative_1d, thresholder):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    model = SR3(thresholder=thresholder)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize("thresholder", ["L0", "l1"])
def test_constrained_sr3_prox_functions(data_derivative_1d, thresholder):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    model = ConstrainedSR3(thresholder=thresholder)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize("thresholder", ["weighted_l0", "weighted_l1"])
def test_weighted_prox_functions(data_derivative_1d, thresholder):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    thresholds = np.ones((x_dot.shape[0], 1))
    model = ConstrainedSR3(thresholder=thresholder, thresholds=thresholds)
    check_is_fitted(model)


def test_unbias(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    optimizer_biased = SINDyOptimizer(
        STLSQ(threshold=0.01, alpha=0.1, max_iter=1), unbias=False
    )
    optimizer_biased.fit(x, x_dot)

    optimizer_unbiased = SINDyOptimizer(
        STLSQ(threshold=0.01, alpha=0.1, max_iter=1), unbias=True
    )
    optimizer_unbiased.fit(x, x_dot)

    assert (
        norm(optimizer_biased.coef_ - optimizer_unbiased.coef_)
        / norm(optimizer_unbiased.coef_)
        > 1e-9
    )


def test_unbias_external(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    optimizer_biased = SINDyOptimizer(
        Lasso(alpha=0.1, fit_intercept=False, max_iter=1), unbias=False
    )
    optimizer_biased.fit(x, x_dot)

    optimizer_unbiased = SINDyOptimizer(
        Lasso(alpha=0.1, fit_intercept=False, max_iter=1), unbias=True
    )
    optimizer_unbiased.fit(x, x_dot)

    assert (
        norm(optimizer_biased.coef_ - optimizer_unbiased.coef_)
        / (norm(optimizer_unbiased.coef_) + 1e-5)
        > 1e-9
    )


def test_initial_guess(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    initial_guess = np.ones_like(x)

    model = STLSQ(initial_guess=initial_guess)
    model.fit(x, x_dot)
    check_is_fitted(model)

    model = SR3(initial_guess=initial_guess)
    model.fit(x, x_dot)
    check_is_fitted(model)

    model = ConstrainedSR3(initial_guess=initial_guess)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "optimizer", [STLSQ(max_iter=1), SR3(max_iter=1), ConstrainedSR3(max_iter=1)],
)
def test_fit_warn(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    with pytest.warns(ConvergenceWarning):
        optimizer.fit(x, x_dot)
