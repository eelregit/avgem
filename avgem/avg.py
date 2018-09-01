import numpy
from scipy.interpolate import make_interp_spline


def _avg(x, y, X, k, axis):
    """interpolate, integrate, and diff.
    """

    y_spl = make_interp_spline(x, y, k=k, axis=axis)

    Y_cum = y_spl.antiderivative()(X)

    Y = numpy.diff(Y_cum, axis=axis)

    return Y

def avg(x, y, X, w=None, wp=None, k=3, axis=-1):
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
    >>> x = numpy.arange(10.)
    >>> y = x
    >>> X = x[::3]
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

    assert all(numpy.diff(x) > 0), "must be strictly increasing"
    assert all(numpy.diff(X) > 0), "must be strictly increasing"

    y = numpy.asarray(y)

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
