from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.exceptions import NotFittedError

from sindy.differentiation import differentiation_methods
from sindy.optimizers import STLSQ
from sindy.utils.base import equation


class SINDy(BaseEstimator):
    def __init__(
            self,
            optimizer=STLSQ(),
            feature_library=PolynomialFeatures(),
            differentiation_method=differentiation_methods.centered_difference,
            feature_names=None,
            n_jobs=1
    ):
        self.optimizer = optimizer
        self.feature_library = feature_library
        self.differentiation_method = differentiation_method
        self.feature_names = feature_names
        self.n_jobs = n_jobs

    # For now just write this how we want it to look and then write simplest
    # version of the supporting code possible
    def fit(self, x, t=1, x_dot=None):

        # Perform checks on input
        check_array(x, ensure_2d=False)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # TODO: check t

        if x_dot is None:
            x_dot = self.differentiation_method(x, t)
        else:
            check_array(x_dot, ensure_2d=False)

        steps = [
            ("features", self.feature_library),
            ("model", self.optimizer),
        ]
        self.model = MultiOutputRegressor(Pipeline(steps), n_jobs=self.n_jobs)

        self.model.fit(x, x_dot)

        self.n_input_features_ = (
            self.model.estimators_[0].steps[0][1].n_input_features_
        )
        self.n_output_features_ = (
            self.model.estimators_[0].steps[0][1].n_output_features_
        )

        return self

    def predict(self, x):
        check_array(x)
        if hasattr(self, 'model'):
            check_is_fitted(
                self.model.estimators_[0].steps[-1][1],
                'coef_'
            )
            return self.model.predict(x)
        else:
            raise NotFittedError(
                "SINDy model must be fit before predict can be called"
            )

    def equations(self, precision=3):
        if hasattr(self, 'model'):
            check_is_fitted(
                self.model.estimators_[0].steps[-1][1],
                'coef_'
            )
            feature_names = (
                self.model.estimators_[0].steps[0][1].get_feature_names(
                    input_features=self.feature_names
                )
            )
            return [
                equation(
                    est,
                    input_features=feature_names,
                    precision=precision
                ) for est in self.model.estimators_
            ]
        else:
            raise NotFittedError(
                "SINDy model must be fit before equations can be called"
            )

    def score(self, x, y=None, multioutput="uniform_average"):
        if y is not None:
            x_dot = y
        else:
            x_dot = self.derivative.transform(x)
        return r2_score(self.model.predict(x), x_dot, multioutput=multioutput)

    @property
    def complexity(self):
        return sum(
            est.steps[1][1].complexity for est in self.model.estimators_
        )

    def differentiate(self, x, t=1):
        return self.differentiation_method(x, t)

    def coefficients(self):
        """Return a list of the coefficients learned by SINDy model"""
        if hasattr(self, 'model'):
            check_is_fitted(
                self.model.estimators_[0].steps[-1][1],
                'coef_'
            )
            return self.model.estimators_[0].steps[-1][1].coef_
        else:
            raise NotFittedError(
                "SINDy model must be fit before coefficients can be called"
            )

    def get_feature_names(self):
        """Return a list of names of features used by SINDy model"""
        if hasattr(self, 'model'):
            return self.model.estimators_[0].steps[0][1].get_feature_names(
                input_features=self.feature_names
            )
        else:
            raise NotFittedError(
                "SINDy model must be fit before get_feature_names can be called"
            )

    # TODO
    def simulate(self, x0, t0, t):
        """
        Simulate forward in time from given initial conditions
        """
