"""The multi-dual number of the noise-bias probe (math.hnumber).

Its arithmetic is checked against the hyper-dual rules written out in
numpy over the layout's tables, and against finite differences of each
function's value along the slots.
"""

import anacal
import numpy as np
import pytest

H = anacal.math.hnumber
NAMES = list(H.names)
NSLOT = len(NAMES)
DIRS = NAMES[1:11]
PAIRS = [tuple(n.split("_")) for n in NAMES[11:]]


def rand(rng, value=None):
    a = rng.normal(size=NSLOT)
    if value is not None:
        a[0] = value
    return a


def product(a, b):
    """The hyper-dual product of two slot arrays."""
    out = np.zeros(NSLOT)
    out[0] = a[0] * b[0]
    for k, _ in enumerate(DIRS, start=1):
        out[k] = a[k] * b[0] + a[0] * b[k]
    for k, (p, q) in enumerate(PAIRS, start=1 + len(DIRS)):
        ip, iq = NAMES.index(p), NAMES.index(q)
        out[k] = a[k] * b[0] + a[0] * b[k] + a[ip] * b[iq] + a[iq] * b[ip]
    return out


def chain(a, f0, f1, f2):
    """f(a) by the chain rule from f, f' and f'' at the value."""
    out = np.zeros(NSLOT)
    out[0] = f0
    out[1:11] = f1 * a[1:11]
    for k, (p, q) in enumerate(PAIRS, start=1 + len(DIRS)):
        ip, iq = NAMES.index(p), NAMES.index(q)
        out[k] = f1 * a[k] + f2 * a[ip] * a[iq]
    return out


def test_layout():
    assert NAMES[:5] == ["v", "g1", "g2", "x1", "x2"]
    assert NAMES[5:11] == ["u", "ug1", "ug2", "ux1", "ux2", "ur"]
    assert PAIRS == [
        ("u", "ug1"),
        ("u", "ug2"),
        ("u", "ux1"),
        ("u", "ux2"),
        ("u", "u"),
        ("ur", "ur"),
    ]
    assert anacal.probe.NSLOT == NSLOT == 17


def test_construction():
    rng = np.random.default_rng(1)
    a = rand(rng)
    h = H(a)
    np.testing.assert_array_equal(h.to_array(), a)
    for k, name in enumerate(NAMES):
        assert getattr(h, name) == a[k]
    c = H(3.0)
    np.testing.assert_array_equal(c.to_array(), [3.0] + [0.0] * (NSLOT - 1))
    assert H().to_array().sum() == 0.0
    assert "hnumber" in repr(h)


def test_arithmetic_rules():
    rng = np.random.default_rng(2)
    a, b = rand(rng, 1.7), rand(rng, -0.6)
    ha, hb = H(a), H(b)
    np.testing.assert_allclose((ha + hb).to_array(), a + b)
    np.testing.assert_allclose((ha - hb).to_array(), a - b)
    np.testing.assert_allclose((-ha).to_array(), -a)
    np.testing.assert_allclose((ha * hb).to_array(), product(a, b))
    np.testing.assert_allclose((ha * 2.5).to_array(), a * 2.5)
    np.testing.assert_allclose((2.5 * ha).to_array(), a * 2.5)
    np.testing.assert_allclose((ha / 2.0).to_array(), a / 2.0)
    plus = a.copy()
    plus[0] += 1.5
    np.testing.assert_allclose((ha + 1.5).to_array(), plus)
    np.testing.assert_allclose((1.5 + ha).to_array(), plus)
    minus = -a
    minus[0] += 1.5
    np.testing.assert_allclose((1.5 - ha).to_array(), minus)
    # division is the product with the reciprocal
    inv = 1.0 / b[0]
    recip = chain(b, inv, -(inv**2), 2.0 * inv**3)
    np.testing.assert_allclose((ha / hb).to_array(), product(a, recip))
    np.testing.assert_allclose((3.0 / hb).to_array(), 3.0 * recip)


@pytest.mark.parametrize(
    "name, f",
    [
        ("exp", (np.exp, np.exp, np.exp)),
        ("log", (np.log, lambda x: 1 / x, lambda x: -1 / x**2)),
        (
            "sqrt",
            (np.sqrt, lambda x: 0.5 / np.sqrt(x), lambda x: -0.25 / x**1.5),
        ),
        (
            "tanh",
            (
                np.tanh,
                lambda x: 1 - np.tanh(x) ** 2,
                lambda x: -2 * np.tanh(x) * (1 - np.tanh(x) ** 2),
            ),
        ),
    ],
)
def test_functions(name, f):
    rng = np.random.default_rng(3)
    a = rand(rng, 1.3)
    f0, f1, f2 = (g(a[0]) for g in f)
    got = getattr(anacal.math, name)(H(a)).to_array()
    np.testing.assert_allclose(got, chain(a, f0, f1, f2), rtol=1e-12)


def test_pow():
    rng = np.random.default_rng(4)
    a = rand(rng, 0.8)
    for n in (2.0, 3.0, -0.5):
        got = anacal.math.pow(H(a), n).to_array()
        want = chain(
            a, a[0] ** n, n * a[0] ** (n - 1), n * (n - 1) * a[0] ** (n - 2)
        )
        np.testing.assert_allclose(got, want, rtol=1e-12)


def test_finite_differences():
    """
    Every slot is a derivative: for a function of one number whose slots
    are the seeds, f(x) along direction d is f'(v) x_d and along a pair
    f''(v) x_p x_q + f'(v) x_pq, which finite differences in the value
    reproduce.
    """
    rng = np.random.default_rng(5)
    a = rand(rng, 1.4)
    h = 1e-4

    def f(x):
        return anacal.math.exp(x * 0.3) * (x * x - 2.0 / x) + 1.0 / (x + 3.0)

    y = f(H(a)).to_array()

    def value(dv):
        b = a.copy()
        b[0] += dv
        return f(H(b)).to_array()[0]

    f1 = (value(h) - value(-h)) / (2 * h)
    f2 = (value(h) - 2 * value(0.0) + value(-h)) / h**2
    np.testing.assert_allclose(y[1:11], f1 * a[1:11], rtol=1e-6)
    for k, (p, q) in enumerate(PAIRS, start=11):
        ip, iq = NAMES.index(p), NAMES.index(q)
        np.testing.assert_allclose(
            y[k], f1 * a[k] + f2 * a[ip] * a[iq], rtol=1e-5
        )


def test_shift_reference():
    a = np.arange(1.0, NSLOT + 1)
    h = H(a)
    dx1, dx2 = 0.5, -0.25
    h.shift_reference(dx1, dx2)
    got = h.to_array()
    want = a.copy()
    i = NAMES.index
    want[i("g1")] += dx1 * a[i("x1")] - dx2 * a[i("x2")]
    want[i("g2")] += dx1 * a[i("x2")] + dx2 * a[i("x1")]
    want[i("ug1")] += dx1 * a[i("ux1")] - dx2 * a[i("ux2")]
    want[i("ug2")] += dx1 * a[i("ux2")] + dx2 * a[i("ux1")]
    want[i("u_ug1")] += dx1 * a[i("u_ux1")] - dx2 * a[i("u_ux2")]
    want[i("u_ug2")] += dx1 * a[i("u_ux2")] + dx2 * a[i("u_ux1")]
    np.testing.assert_allclose(got, want)


def test_matches_anacal_hyperdual():
    """Against the reference implementation, where it is installed."""
    md = pytest.importorskip("anacal_hyperdual.multidual")
    P = md.make_multidual(md.PROBE_SPEC)
    assert list(P.FIELDS) == NAMES
    rng = np.random.default_rng(6)
    a, b = rand(rng, 2.1), rand(rng, 0.9)
    pa, pb = P(a), P(b)
    for ours, ref in (
        (H(a) * H(b), pa * pb),
        (H(a) / H(b), pa / pb),
        (H(a) - 2.0 * H(b) + 0.5, pa - 2.0 * pb + 0.5),
    ):
        np.testing.assert_allclose(ours.to_array(), ref.to_array(), rtol=1e-13)
    h = H(a)
    h.shift_reference(0.3, -0.2)
    np.testing.assert_allclose(
        h.to_array(), P.shift_reference(pa, 0.3, -0.2).to_array(), rtol=1e-13
    )
