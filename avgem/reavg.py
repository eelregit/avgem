import numpy
from scipy.interpolate import make_interp_spline


def _reavg(X0, Y0, X1, k, axis):
    """cumsum, interpolate, and diff.
    """

    Y0_cum = Y0.cumsum(axis=axis)
    if axis == -1: axis += Y0.ndim
    _Y0_cum = numpy.zeros(Y0.shape[:axis] + (1,) + Y0.shape[axis+1:])
    Y0_cum = numpy.concatenate((_Y0_cum, Y0_cum), axis=axis)

    Y1_cum = make_interp_spline(X0, Y0_cum, k=k, axis=axis)(X1)

    Y1 = numpy.diff(Y1_cum, axis=axis)

    return Y1

def reavg(X0, Y0, X1, W0=None, W1=None, k=3, axis=0):
    r"""Re-average an averaged function (with averaged weight) to different bins.

    .. math:: Y_{1,i} = \frac{Z(X_{1,i+1}) - Z(X_{1,i})}{W_{1,i}}

    where :math:`Z(x)` is the spline interpolation of the cumulative sum

    .. math:: Z(X_{0,i}) = \sum_{j=0}^{i-1} Y_{0,j} W_{0,j}

    If not given, :math:`W_1` is determined from :math:`W_0`

    .. math:: W_{1,i} = V(X_{1,i+1}) - V(X_{1,i})

    where :math:`V(x)` is the spline interpolation of the cumulative sum

    .. math:: V(X_{0,i}) = \sum_{j=0}^{i-1} W_{0,j}

    Parameters
    ----------
    X0 : (B0+1,) array_like
        input bin edges
    Y0 : (..., B0, ...) array_like
        averaged function in `X0` along `axis`
    X1 : (B1+1,) array_like
        output bin edges
    W0 : (B0,) array_like, optional
        averaged input weight. Default is linear in `X0`
    W1 : (B1,) array_like, optional
        re-averaged output weight. Default is determined from `W0`
    k : int, optional
        degree of spline for interpolation and integration
    axis : int, optional
        axis of `Y0` along which to re-average

    Returns
    -------
    Y1 : (..., B1, ...) ndarray
        re-averaged function in `X1` along `axis`
    W1 : (B1,) ndarray
        re-averaged weight

    Examples
    --------
    >>> x = numpy.arange(10.)  # array([0., 1., ... 9.])
    >>> y = x
    >>> X0 = x[::3]  # array([0., 3., 6., 9.])
    >>> Y0 = numpy.array([1.5, 4.5, 7.5])  # Y0, W0 = avg(x, y, X0)
    >>> W0 = numpy.array([3., 3., 3.])

    >>> X1 = x[::2]  # array([0., 2., 4., 6., 8.])
    >>> Y1, W1 = reavg(X0, Y0, X1)
    >>> Y1
    array([1., 3., 5., 7.])
    >>> W1
    array([2., 2., 2., 2.])

    >>> Y1, W1 = reavg(X0, Y0, X1, W0=W0, W1=W1/2)
    >>> Y1
    array([ 2.,  6., 10., 14.])
    >>> W1
    array([1., 1., 1., 1.])
    """

    X0 = numpy.asarray(X0)
    X1 = numpy.asarray(X1)
    Y0 = numpy.asarray(Y0)

    if X0.ndim != 1 or any(X0[:-1] >= X0[1:]):
        raise ValueError("X0 must be 1-D and strictly increasing")
    if X1.ndim != 1 or any(X1[:-1] >= X1[1:]):
        raise ValueError("X1 must be 1-D and strictly increasing")

    if not -Y0.ndim <= axis < Y0.ndim:
        raise ValueError("axis {} is out of bounds".format(axis))
    if axis < 0:
        axis += Y0.ndim

    to_axis = [1] * Y0.ndim
    to_axis[axis] = -1

    W0 = numpy.diff(X0) if W0 is None else numpy.asarray(W0)
    W1 = _reavg(X0, W0, X1, k, axis) if W1 is None else numpy.asarray(W1)
    W0 = W0.reshape(to_axis)
    W1 = W1.reshape(to_axis)

    Y1 = _reavg(X0, Y0 * W0, X1, k, axis)
    Y1 /= W1

    return Y1, W1.squeeze()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
