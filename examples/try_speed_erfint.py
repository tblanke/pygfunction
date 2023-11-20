# Current implementation
from timeit import timeit

import numpy as np
# from scipy.special import erf, erfinv
from scipy.special import erf, erfinv

def erfint_0(x):
    """
    Integral of the error function.

    Parameters
    ----------
    x : float or array
        Argument.

    Returns
    -------
    float or array
        Integral of the error function.

    """
    return x * erf(x) - 1.0 / np.sqrt(np.pi) * (1.0 - np.exp(-x *x))


# Proposed implementation
def erfint_1(x, tol=1e-8):
    """
    Integral of the error function.



    Parameters
    ----------
    x : float or array
        Argument.
    tol : float, optional
        Relative tolerance on the value of erfint.
        Default is 1e-8.

    Returns
    -------
    float or array
        Integral of the error function.

    """
    # Determine the threshold for tolerance
    x_tol = np.maximum(erfinv(1. - tol), np.sqrt(-np.log(tol)))
    # Apply the approximation to all x
    abs_x = np.abs(x)
    y = abs_x - 1. / np.sqrt(np.pi)
    # Evaluate erfint for x < x_tol
    idx = np.less(abs_x, x_tol)
    x_low = abs_x[idx]
    y[idx] = x_low * erf(x_low) - (1.0 - np.exp(-x_low * x_low)) / np.sqrt(np.pi)
    return y


def main():
    num = 100
    x_tol = np.maximum(erfinv(1. - 1e-8), np.sqrt(-np.log(1e-8)))
    x = np.random.random(num)
    print(timeit("erfint_0(x)", globals={"erfint_0": erfint_0, "x": x}))
    print(timeit("erfint_1(x)", globals={"erfint_1": erfint_1, "x": x}))
    x = x + x_tol - 0.75
    print(timeit("erfint_0(x)", globals={"erfint_0": erfint_0, "x": x}))
    print(timeit("erfint_1(x)", globals={"erfint_1": erfint_1, "x": x}))
    x = x + 0.25
    print(timeit("erfint_0(x)", globals={"erfint_0": erfint_0, "x": x}))
    print(timeit("erfint_1(x)", globals={"erfint_1": erfint_1, "x": x}))
    x = x + 0.25
    print(timeit("erfint_0(x)", globals={"erfint_0": erfint_0, "x": x}))
    print(timeit("erfint_1(x)", globals={"erfint_1": erfint_1, "x": x}))

# Main function
if __name__ == '__main__':
    main()