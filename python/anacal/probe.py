"""
The shear estimator of the noise-bias probe.

``anacal.probe.process_cell`` (C++) detects and measures a cell with the
probe: every quantity carries, besides the image's shear and translation
responses, its responses along a pure-noise image (``u``, ``ug1``, ...)
and the second-order slots that the noise-bias correction of a
response-calibrated estimator needs.  Over the rows of its catalog the
estimator is

    g1 = sum(S1 - C1) / sum(R1 - B1)

with ``S1 = wsel e1`` the summand, ``R1 = S1_g1`` its shear response,
``B1 = S1_u_ug1`` the noise-bias term of the response (Stein's identity:
the mean of the noise times the derivative of the response along the
noise) and ``C1 = (S1_u_u - S1_ur_ur) / 4`` the additive term, from the
rotated noise.  No noise was added to the image, so the estimator keeps
the image's own noise, and the terms are exact in expectation for
Gaussian noise of any covariance the realization has.

A selection made after the fact, a smooth step on a catalog quantity,
is carried into the sums by ``select``: the step's slots follow by the
chain rule from the quantity's, the product with the summand by the
product rule, so the selection's response and noise-bias terms are in
the corrected estimate.  Every quantity is written with all its slots
for that purpose.
"""

import numpy as np

from ._anacal import probe as _probe
from ._anacal.math import hnumber

process_cell = _probe.process_cell
prepare_planes = _probe.prepare_planes
NSLOT = _probe.NSLOT

__all__ = [
    "process_cell",
    "prepare_planes",
    "sums",
    "shear",
    "select",
    "slots",
    "product",
    "chain",
    "ssfunc2",
    "NAMES",
    "NSLOT",
]

NAMES = list(hnumber.names)
_NDIR = 10
# the pairs of the layout, as index pairs into the slot array
_PAIRS = [
    tuple(NAMES.index(n) for n in name.split("_"))
    for name in NAMES[1 + _NDIR :]
]


def slots(cat, stem):
    """
    The slots of a catalog quantity as one array.

    Parameters
    ----------
    cat : ndarray
        A catalog of ``process_cell``.
    stem : str
        The quantity, e.g. ``"m0"``; its columns are ``stem`` and
        ``stem_<slot>``.

    Returns
    -------
    ndarray, (n, NSLOT)
        The slots of every row, value first, in ``hnumber.names`` order.
    """
    return np.stack(
        [cat[stem if n == "v" else f"{stem}_{n}"] for n in NAMES], axis=1
    )


def product(a, b):
    """
    The product of two slot arrays, row by row.

    The hyper-dual product rule over the layout's pairs, as
    ``math.hnumber.__mul__``.

    Parameters
    ----------
    a, b : ndarray, (n, NSLOT)
        The factors' slots.

    Returns
    -------
    ndarray, (n, NSLOT)
        The product's slots.
    """
    out = np.empty_like(a)
    av, bv = a[:, :1], b[:, :1]
    out[:, 0] = a[:, 0] * b[:, 0]
    out[:, 1 : 1 + _NDIR] = a[:, 1 : 1 + _NDIR] * bv + av * b[:, 1 : 1 + _NDIR]
    for k, (p, q) in enumerate(_PAIRS, start=1 + _NDIR):
        out[:, k] = (
            a[:, k] * b[:, 0]
            + a[:, 0] * b[:, k]
            + a[:, p] * b[:, q]
            + a[:, q] * b[:, p]
        )
    return out


def chain(a, f0, f1, f2):
    """
    A function of a slot array by the chain rule, row by row.

    Parameters
    ----------
    a : ndarray, (n, NSLOT)
        The argument's slots.
    f0, f1, f2 : ndarray, (n,)
        The function and its first two derivatives at the values.

    Returns
    -------
    ndarray, (n, NSLOT)
        The result's slots: ``f1 a_d`` for a direction, ``f1 a_pq + f2
        a_p a_q`` for a pair.
    """
    out = np.empty_like(a)
    out[:, 0] = f0
    out[:, 1 : 1 + _NDIR] = f1[:, None] * a[:, 1 : 1 + _NDIR]
    for k, (p, q) in enumerate(_PAIRS, start=1 + _NDIR):
        out[:, k] = f1 * a[:, k] + f2 * a[:, p] * a[:, q]
    return out


def ssfunc2(a, mu, sigma):
    """
    The C2 smooth step of a slot array, row by row.

    AnaCal's quintic step (``math.ssfunc2``): with ``t = (x - mu) / (2
    sigma) + 1/2`` clipped to [0, 1], ``s = 6 t^5 - 15 t^4 + 10 t^3``,
    rising from 0 at ``mu - sigma`` to 1 at ``mu + sigma``.  Its second
    derivative is continuous, so the noise-bias terms of a selection
    are continuous in the image.

    Parameters
    ----------
    a : ndarray, (n, NSLOT)
        The argument's slots.
    mu, sigma : float
        The center and half-width of the ramp.

    Returns
    -------
    ndarray, (n, NSLOT)
        The step's slots.
    """
    w = 2.0 * sigma
    t = np.clip((a[:, 0] - mu) / w + 0.5, 0.0, 1.0)
    ramp = (t > 0.0) & (t < 1.0)
    f0 = t**3 * (6.0 * t * t - 15.0 * t + 10.0)
    f1 = np.where(ramp, 30.0 * t * t * (1.0 - t) ** 2 / w, 0.0)
    f2 = np.where(ramp, 60.0 * t * (1.0 - t) * (1.0 - 2.0 * t) / w**2, 0.0)
    return chain(a, f0, f1, f2)


def select(cat, column, mu, sigma, stem="S"):
    """
    A smooth selection on a catalog quantity, carried into the sums.

    The summands ``S1``, ``S2`` are multiplied by the smooth step
    ``ssfunc2(column, mu, sigma)`` with the product rule, so the
    estimator read from the result (``sums``, ``shear``) includes the
    selection's shear response and its noise-bias terms: the correction
    for the selection effect of the cut.

    Parameters
    ----------
    cat : ndarray
        A catalog of ``process_cell``.
    column : str
        The quantity cut on, e.g. ``"m0"``.
    mu, sigma : float
        The ramp: zero weight below ``mu - sigma``, full above ``mu +
        sigma``.
    stem : str, optional
        The summands' column stem.

    Returns
    -------
    ndarray
        A copy of ``cat`` with the selected summands in place of
        ``stem1``, ``stem2`` and their slots.
    """
    out = cat.copy()
    w = ssfunc2(slots(cat, column), mu, sigma)
    for a in (1, 2):
        prod = product(w, slots(cat, f"{stem}{a}"))
        for k, n in enumerate(NAMES):
            out[f"{stem}{a}" if n == "v" else f"{stem}{a}_{n}"] = prod[:, k]
    return out


def sums(cat, stem="S"):
    """
    The estimator's sums over a catalog, per component.

    Parameters
    ----------
    cat : ndarray
        A catalog of ``process_cell``, or several concatenated.
    stem : str, optional
        The summand's column stem, ``"S"`` for ``S1``/``S2``.

    Returns
    -------
    dict
        For each component ``a`` in (1, 2): ``N{a}`` the sum of the
        values, ``R{a}`` of the shear responses, ``B{a}`` of the
        noise-bias terms and ``C{a}`` of the additive terms.
    """
    out = {}
    for a in (1, 2):
        col = f"{stem}{a}"
        out[f"N{a}"] = float(np.sum(cat[col]))
        out[f"R{a}"] = float(np.sum(cat[f"{col}_g{a}"]))
        out[f"B{a}"] = float(np.sum(cat[f"{col}_u_ug{a}"]))
        out[f"C{a}"] = 0.25 * float(
            np.sum(cat[f"{col}_u_u"]) - np.sum(cat[f"{col}_ur_ur"])
        )
    return out


def shear(cat, correct=True):
    """
    The shear estimate from a catalog.

    Parameters
    ----------
    cat : ndarray
        A catalog of ``process_cell``, or several concatenated.
    correct : bool, optional
        Apply the noise-bias terms; False gives the uncorrected
        ``sum S / sum R``.

    Returns
    -------
    g1, g2 : float
        The shear estimate.
    """
    s = sums(cat)
    out = []
    for a in (1, 2):
        if correct:
            num = s[f"N{a}"] - s[f"C{a}"]
            den = s[f"R{a}"] - s[f"B{a}"]
        else:
            num = s[f"N{a}"]
            den = s[f"R{a}"]
        out.append(num / den)
    return tuple(out)
