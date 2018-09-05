import numpy
from scipy.interpolate import make_interp_spline


def _avg(x, y, X, k, axis):
    """interpolate, integrate, and diff.
    """

    y_spl = make_interp_spline(x, y, k=k, axis=axis)

    Y_cum = y_spl.antiderivative()(X)

    Y = numpy.diff(Y_cum, axis=axis)

    return Y

def avg(x, y, X, w=None, wp=None, k=3, axis=0):
    r"""Average a function (with weight) to bins.

    .. math:: Y_i = \frac{\int_{X_i}^{X_{i+1}} y(x) w'(x) dx}
                            {\int_{X_i}^{X_{i+1}} w(x) dx}

    Parameters
    ----------
    x : (N,) array_like
        argument
    y : (..., N, ...) array_like
        function of `x` along `axis`
    X : (B+1,) array_like
        bin edges
    w : (N,) array_like, optional
        weight for both the denominator and the numerator when `wp` is `None`,
        or only the denominator otherwise. Default is 1
    wp : (N,) array_like, optional
        weight for the numerator. Default is `w`
    k : int, optional
        degree of spline for interpolation and integration
    axis : int, optional
        axis of `y` along which to average

    Returns
    -------
    Y : (..., B, ...) ndarray
        averaged function in `X` along `axis`
    W : (B,) ndarray
        averaged weight

    Examples
    --------
    >>> x = numpy.arange(10.)  # array([0., 1., ... 9.])
    >>> y = x
    >>> X = x[::3]  # array([0., 3., 6., 9.])
    >>> Y, W = avg(x, y, X)
    >>> Y
    array([1.5, 4.5, 7.5])
    >>> W
    array([3., 3., 3.])

    >>> w = x**2
    >>> wp = x
    >>> Y, W = avg(x, y, X, w=w, wp=wp)
    >>> Y
    array([1., 1., 1.])
    >>> W
    array([  9.,  63., 171.])
    """

    x = numpy.asarray(x)
    X = numpy.asarray(X)
    y = numpy.asarray(y)

    if x.ndim != 1 or any(x[:-1] >= x[1:]):
        raise ValueError("x must be 1-D and strictly increasing")
    if X.ndim != 1 or any(X[:-1] >= X[1:]):
        raise ValueError("X must be 1-D and strictly increasing")

    if not -y.ndim <= axis < y.ndim:
        raise ValueError("axis {} is out of bounds".format(axis))
    if axis < 0:
        axis += y.ndim

    to_axis = [1] * y.ndim
    to_axis[axis] = -1

    w = numpy.ones_like(x) if w is None else numpy.asarray(w)
    wp = w if wp is None else numpy.asarray(wp)
    w = w.reshape(to_axis)
    wp = wp.reshape(to_axis)

    W = _avg(x, w, X, k, axis)
    Y = _avg(x, y * wp, X, k, axis)
    Y /= W

    return Y, W.squeeze()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
