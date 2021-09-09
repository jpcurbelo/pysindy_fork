from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

from numpy import asarray
from numpy import empty
from numpy import hstack
from numpy import linspace
from numpy import ones
from numpy import reshape
from numpy import shape
from numpy import zeros
from numpy.random import uniform
from scipy.integrate import trapezoid
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from pysindy.differentiation import FiniteDifference

# from scipy.special import binom


class PDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable (but not their derivatives)

    derivative_order : int, optional (default 0)
        Order of derivative to take on each input variable, max 3

    spatial_grid : np.ndarray, optional (default None)
        The spatial grid for computing derivatives

    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names list must be the
        same length as library_functions. If no list of function names is
        provided, defaults to using :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`.

    interaction_only : boolean, optional (default True)
        Whether to omit self-interaction terms.
        If True, function evaulations of the form :math:`f(x,x)` and :math:`f(x,y,x)`
        will be omitted, but those of the form :math:`f(x,y)` and :math:`f(x,y,z)`
        will be included.
        If False, all combinations will be included.

    include_bias : boolean, optional (default True)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    is_uniform : boolean, optional (default True)
        If True, assume the grid is uniform in all spatial directions, so
        can use uniform grid spacing for the derivative calculations.

    weak_form : boolean, optional (default False)
        If True, uses the weak/integral form of SINDy, requiring some extra
        parameters.

    domain_centers : np.ndarray, optional (default None)
        List of domain centers, corresponding to subdomain squares of length
        Hx and height Hy. If weak_form is True but domain_centers is not
        specified, defaults to size (100, n_spatial_dims).

    Hx : float, optional (default None)
        Length of the square subdomains. If weak_form is True but Hx is not
        specified, defaults to Hx = Lx / 20, where Lx is the length of the
        full domain.

    Hx : float, optional (default None)
        Height of the square subdomains. If weak_form is True but Hy is not
        specified, defaults to Hy = Ly / 20, where Lx is the length of the
        full domain.

    p : int, optional (default 4)
        Positive integer to define the polynomial degree of the spatial weights
        used for weak/integral SINDy.

    Attributes
    ----------
    functions : list of functions
        Mathematical library functions to be applied to each input feature.

    function_names : list of functions
        Functions for generating string representations of each library
        function.

    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import PDELibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib = PDELibrary(library_functions=functions).fit(x)
    >>> lib.transform(x)
    array([[ 1.        ,  0.36787944, -0.84147098],
           [ 2.71828183,  1.        ,  0.84147098],
           [ 7.3890561 ,  0.36787944,  0.84147098]])
    >>> lib.get_feature_names()
    ['f0(x0)', 'f0(x1)', 'f1(x0,x1)']
    """

    def __init__(
        self,
        library_functions,
        derivative_order=0,
        spatial_grid=None,
        function_names=None,
        interaction_only=True,
        include_bias=True,
        is_uniform=False,
        weak_form=False,
        domain_centers=None,
        Hx=None,
        Hy=None,
        p=4,
    ):
        super(PDELibrary, self).__init__()
        self.derivative_order = derivative_order
        self.spatial_grid = spatial_grid
        self.functions = library_functions
        self.function_names = function_names
        self.include_bias = include_bias
        self.is_uniform = is_uniform
        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if derivative_order < 0 or derivative_order > 4:
            raise ValueError("The derivative order must be 1, 2, 3, or 4")
        if (spatial_grid is not None and derivative_order == 0) or (
            spatial_grid is None and derivative_order != 0
        ):
            raise ValueError(
                "Spatial grid and the derivative order must be "
                "defined at the same time"
            )
        if spatial_grid is not None and (
            (len(spatial_grid.shape) != 1)
            and (len(spatial_grid.shape) != 3)
            and (len(spatial_grid.shape) != 4)
        ):
            raise ValueError("Spatial grid size is incorrect")
        self.interaction_only = interaction_only
        if spatial_grid is not None:
            if len(spatial_grid.shape) == 1:
                self.num_derivs = derivative_order
            elif len(spatial_grid.shape) == 3:
                num_derivs = 2
                for i in range(2, derivative_order + 1):
                    num_derivs += i + 1
                self.num_derivs = num_derivs

        # weak form checks now
        if weak_form and spatial_grid is None:
            raise ValueError("Weak form requires user to pass a spatial grid.")
        self.weak_form = weak_form
        self.p = p
        if weak_form and spatial_grid is not None:
            if p < 0:
                raise ValueError("Poly degree of the spatial weights must be > 0")
            if self.Hx is not None:
                if self.Hx <= 0:
                    raise ValueError("Hx must be a positive float")
                if len(spatial_grid.shape) == 1:
                    x1 = spatial_grid[0]
                    x2 = spatial_grid[-1]
                if len(spatial_grid.shape) == 3:
                    x1 = spatial_grid[0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0]
                if self.Hx >= (x2 - x1):
                    raise ValueError("Hx is bigger than the full domain")
            else:
                if len(spatial_grid.shape) == 1:
                    x1 = spatial_grid[0]
                    x2 = spatial_grid[-1]
                    Lx = x2 - x1
                    self.Hx = Lx / 20.0
                if len(spatial_grid.shape) == 3:
                    x1 = spatial_grid[0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0]
                    Lx = x2 - x1
                    self.Hx = Lx / 20.0
            if self.Hy is not None and len(spatial_grid.shape) == 3:
                if self.Hy <= 0:
                    raise ValueError("Hy must be a positive float")
                y1 = spatial_grid[0, 0, 1]
                y2 = spatial_grid[0, -1, 1]
                if self.Hy >= (y2 - y1):
                    raise ValueError("Hy is bigger than the full domain")
            elif self.Hy is None and len(spatial_grid.shape) == 3:
                y1 = spatial_grid[0, 0, 0]
                y2 = spatial_grid[0, -1, 1]
                Ly = y2 - y1
                self.Hy = Ly / 20.0
            if domain_centers is not None and shape(domain_centers) != 2:
                raise ValueError(
                    "Subdomain center points must have shape "
                    "(n_subdomains, n_spatial_dims)."
                )
            elif shape(domain_centers)[1] != 1 and shape(domain_centers)[1] != 2:
                raise ValueError("Subdomains only supported in 1D or 2D")
            if domain_centers is None:
                self.K = 100
                if len(spatial_grid.shape) == 1:
                    x1 = spatial_grid[0]
                    x2 = spatial_grid[-1]
                    domain_centers = uniform(
                        x1 + self.Hx / 2.0, x2 - self.Hx / 2.0, size=(self.K, 1)
                    )
                if len(spatial_grid.shape) == 3:
                    x1 = spatial_grid[0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0]
                    y1 = spatial_grid[0, 0, 0]
                    y2 = spatial_grid[-1, 0, 0]
                    domain_centersx = uniform(
                        x1 + self.Hx / 2.0, x2 - self.Hx / 2.0, size=(self.K, 1)
                    )
                    domain_centersy = uniform(
                        y1 + self.Hy / 2.0, y2 - self.Hy / 2.0, size=(self.K, 1)
                    )
                    domain_centers = hstack((domain_centersx, domain_centersy))
            else:
                self.K = shape(domain_centers)[0]
            self.domain_centers = domain_centers

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def _smooth_ppoly(self, x, k):
        if shape(x)[1] == 1:
            x_tilde = zeros(shape(x)[0])
            for i in range(shape(x)[0]):
                x_tilde[i] = (x[i] - self.domain_centers[k, 0]) / self.Hx
            return (x_tilde ** 2 - 1) ** self.p
        if shape(x)[1] == 2:
            x_tilde = zeros(shape(x)[0])
            y_tilde = zeros(shape(x)[0])
            for i in range(shape(x)[0]):
                x_tilde[i] = (x[i, 0] - self.domain_centers[k, 0]) / self.Hx
                y_tilde[i] = (x[i, 1] - self.domain_centers[k, 1]) / self.Hy
            return (x_tilde ** 2 - 1) ** self.p * (y_tilde ** 2 - 1) ** self.p

    def _make_2D_derivatives(self, u):
        (num_gridx, num_gridy, num_time, n_features) = u.shape
        u_derivs = zeros((num_gridx, num_gridy, num_time, n_features, self.num_derivs))
        # first derivatives
        for i in range(num_time):
            # u_x
            for kk in range(num_gridy):
                u_derivs[:, kk, i, :, 0] = FiniteDifference(
                    d=1, is_uniform=self.is_uniform
                )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
            # u_y
            for kk in range(num_gridx):
                u_derivs[kk, :, i, :, 1] = FiniteDifference(
                    d=1, is_uniform=self.is_uniform
                )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 2:
            # second derivatives
            for i in range(num_time):
                # u_xx
                for kk in range(num_gridy):
                    u_derivs[:, kk, i, :, 2] = FiniteDifference(
                        d=2, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 3] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 0], self.spatial_grid[kk, :, 1]
                    )
                # u_yy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 4] = FiniteDifference(
                        d=2, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 3:
            # third derivatives
            for i in range(num_time):
                # u_xxx
                for kk in range(num_gridy):
                    u_derivs[:, kk, i, :, 5] = FiniteDifference(
                        d=3, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xxy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 6] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 2], self.spatial_grid[kk, :, 1]
                    )
                # u_xyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 7] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 3], self.spatial_grid[kk, :, 1]
                    )
                # u_yyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 8] = FiniteDifference(
                        d=3, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 4:
            # fourth derivatives
            for i in range(num_time):
                # u_xxxx
                for kk in range(num_gridy):
                    u_derivs[:, kk, i, :, 9] = FiniteDifference(
                        d=4, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xxxy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 10] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 5], self.spatial_grid[kk, :, 1]
                    )
                # u_xxyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 11] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 6], self.spatial_grid[kk, :, 1]
                    )
                # u_xyyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 12] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivs[kk, :, i, :, 7], self.spatial_grid[kk, :, 1]
                    )
                # u_yyyy
                for kk in range(num_gridx):
                    u_derivs[kk, :, i, :, 13] = FiniteDifference(
                        d=4, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])
        return u_derivs

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        check_is_fitted(self)
        if input_features is None:
            input_features = ["x%d" % i for i in range(self.n_input_features_)]
        feature_names = []
        if self.include_bias:
            feature_names.append("1")
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                self.n_input_features_, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )
        for k in range(self.num_derivs):
            for j in range(self.n_input_features_):
                feature_names.append(
                    self.function_names[len(self.functions) + k](input_features[j])
                )
        for k in range(self.num_derivs):
            for i, f in enumerate(self.functions):
                for c in self._combinations(
                    self.n_input_features_,
                    f.__code__.co_argcount,
                    self.interaction_only,
                ):
                    for jj in range(self.n_input_features_):
                        feature_names.append(
                            self.function_names[i](*[input_features[j] for j in c])
                            + self.function_names[len(self.functions) + k](
                                input_features[jj]
                            )
                        )
        return feature_names

    def fit(self, x, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(x).shape
        self.n_input_features_ = n_features

        # Need to compute any derivatives now
        if self.spatial_grid is not None:
            # 1D space
            if len((self.spatial_grid).shape) == 1:
                num_gridpts = (self.spatial_grid).shape[0]
                num_time = n_samples // num_gridpts
                u = reshape(x, (num_gridpts, num_time, n_features), "F")

                if self.weak_form:
                    self.u_derivs = zeros(
                        (self.K, num_time, n_features, self.derivative_order)
                    )
                    for k in range(self.K):
                        xgrid_k = linspace(
                            self.domain_centers[k, 0] - self.Hx / 2.0,
                            self.domain_centers[k, 0] + self.Hx / 2.0,
                            100,
                        )
                        for i in range(num_time):
                            for j in range(self.derivative_order):
                                w_diff = FiniteDifference(
                                    d=j + 1, is_uniform=self.is_uniform
                                )._differentiate(
                                    self._smooth_ppoly(
                                        self.domain_centers, self.domain_centers, k
                                    ),
                                    self.domain_centers[k, 0],
                                )
                                self.u_derivs[k, i, j] = trapezoid(
                                    u[:, i, :] * w_diff, x=xgrid_k
                                )
                else:
                    self.u_derivs = zeros(
                        (num_gridpts, num_time, n_features, self.derivative_order)
                    )
                    for i in range(num_time):
                        for j in range(self.derivative_order):
                            self.u_derivs[:, i, :, j] = FiniteDifference(
                                d=j + 1, is_uniform=self.is_uniform
                            )._differentiate(u[:, i, :], self.spatial_grid)
            # 2D space
            elif len(self.spatial_grid.shape) == 3:
                num_gridx = (self.spatial_grid).shape[0]
                num_gridy = (self.spatial_grid).shape[1]
                num_time = n_samples // num_gridx // num_gridy
                u = reshape(x, (num_gridx, num_gridy, num_time, n_features), "F")
                self.u_derivs = self._make_2D_derivatives(u)

        n_output_features = 0
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )
        if self.spatial_grid is not None:
            # Add the mixed derivative library_terms
            n_output_features += n_output_features * n_features * self.num_derivs
            # Add the pure derivative library terms
            n_output_features += n_features * self.num_derivs
        if self.include_bias:
            n_output_features += 1
        self.n_output_features_ = n_output_features
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(n_features),
                )
            )
        if self.spatial_grid is not None:
            if len((self.spatial_grid).shape) == 1:
                # pure derivative terms
                self.function_names = hstack(
                    (
                        self.function_names,
                        list(
                            map(
                                lambda i: (lambda *x: "".join(x) + "_" + "1" * i),
                                range(1, self.derivative_order + 1),
                            )
                        ),
                    )
                )
            elif len(self.spatial_grid.shape) == 3:
                deriv_strings = [
                    "1",
                    "2",
                    "11",
                    "12",
                    "22",
                    "111",
                    "112",
                    "122",
                    "222",
                    "1111",
                    "1112",
                    "1122",
                    "1222",
                    "2222",
                ]
                # pure derivative terms
                self.function_names = hstack(
                    (
                        self.function_names,
                        list(
                            map(
                                lambda i: (
                                    lambda *x: "".join(x) + "_" + deriv_strings[i]
                                ),
                                range(0, self.num_derivs),
                            )
                        ),
                    )
                )
        return self

    def transform(self, x):
        """Transform data to custom features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of features
            generated from applying the custom functions to the inputs.
        """
        check_is_fitted(self)

        x = check_array(x)

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x shape does not match training shape")

        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        library_idx = 0
        if self.include_bias:
            xp[:, library_idx] = ones(n_samples)
            library_idx += 1
        for f in self.functions:
            for c in self._combinations(
                self.n_input_features_, f.__code__.co_argcount, self.interaction_only
            ):
                xp[:, library_idx] = f(*[x[:, j] for j in c])
                library_idx += 1

        # Need to recompute derivatives because the training
        # and testing data may have different number of time points
        # Need to compute any derivatives now
        if self.spatial_grid is not None:
            if len((self.spatial_grid).shape) == 1:
                num_gridpts = (self.spatial_grid).shape[0]
                num_time = n_samples // num_gridpts
                u = reshape(x, (num_gridpts, num_time, n_features), "F")
                u_derivs = zeros(
                    (num_gridpts, num_time, n_features, self.derivative_order)
                )
                for i in range(num_time):
                    for j in range(self.derivative_order):
                        u_derivs[:, i, :, j] = FiniteDifference(
                            d=j + 1, is_uniform=self.is_uniform
                        )._differentiate(u[:, i, :], self.spatial_grid)
                u_derivs = asarray(
                    reshape(
                        u_derivs,
                        (num_gridpts * num_time, n_features, self.derivative_order),
                        "F",
                    )
                )
            elif len((self.spatial_grid).shape) == 3:
                num_gridx = (self.spatial_grid).shape[0]
                num_gridy = (self.spatial_grid).shape[1]
                num_time = n_samples // num_gridx // num_gridy
                u = reshape(x, (num_gridx, num_gridy, num_time, n_features), "F")
                u_derivs = self._make_2D_derivatives(u)
                u_derivs = asarray(
                    reshape(
                        u_derivs,
                        (num_gridx * num_gridy * num_time, n_features, self.num_derivs),
                        "F",
                    )
                )

            def derivative_function(y):
                return y

            if len((self.spatial_grid).shape) == 1:
                for k in range(self.num_derivs):
                    for kk in range(n_features):
                        xp[:, library_idx] = derivative_function(u_derivs[:, kk, k])
                        library_idx += 1

                # All the mixed derivative/non-derivative terms
                for k in range(self.num_derivs):
                    for kk in range(n_features):
                        for f in self.functions:
                            for c in self._combinations(
                                self.n_input_features_,
                                f.__code__.co_argcount,
                                self.interaction_only,
                            ):
                                xp[:, library_idx] = f(
                                    *[x[:, j] for j in c]
                                ) * derivative_function(u_derivs[:, kk, k])
                                library_idx += 1

            if len((self.spatial_grid).shape) == 3:
                for k in range(self.num_derivs):
                    for kk in range(n_features):
                        xp[:, library_idx] = derivative_function(u_derivs[:, kk, k])
                        library_idx += 1

                # All the mixed derivative/non-derivative terms
                for k in range(self.num_derivs):
                    for kk in range(n_features):
                        for f in self.functions:
                            for c in self._combinations(
                                self.n_input_features_,
                                f.__code__.co_argcount,
                                self.interaction_only,
                            ):
                                xp[:, library_idx] = f(
                                    *[x[:, j] for j in c]
                                ) * derivative_function(u_derivs[:, kk, k])
                                library_idx += 1
        return xp
