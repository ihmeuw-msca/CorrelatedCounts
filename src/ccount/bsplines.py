import numpy as np
import xspline


def spline_design_mat(array, knots_type='frequency', knots_num=3,
                      degree=3, l_linear=True, r_linear=True):
    """

    Args:
        array: (np.array)
        knots_type: (str) optional, one of 'frequency' which places the knots according to quantiles of the data
            or 'domain' which places the knots according to the domain of the data
        knots_num: (int) number of knots
        degree: (int) degree of the spline
        l_linear: (bool) linear tails on the left, optional
        r_linear: (bool) linear tails on the right, optional

    Returns:
        pd.DataFrame
    """
    spline_knots = np.linspace(0, 1, knots_num)
    if knots_type == 'frequency':
        knots = np.quantile(array, spline_knots)
    else:
        knots = array.min() + spline_knots * (array.max() - array.min())
    xs = xspline.XSpline(
        knots=knots,
        degree=degree,
        l_linear=l_linear,
        r_linear=r_linear
    )
    return xs.design_mat(array)[:, 1:]
