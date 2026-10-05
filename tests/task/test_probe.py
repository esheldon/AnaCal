"""Detection and moments with the noise-bias probe (anacal.probe).

On the tiled-stamp field of ``test_detection``: the probe finds the same
pixels as the qnumber detector with the same first-order slots; its
noise slots are the derivatives along the noise image, checked by finite
differences; the estimator recovers the stamp's shear.
"""

import anacal
import numpy as np
import pytest

from ..fixtures import load

SCALE = 0.2
NGAL = 12
VAR = 0.10
KW = dict(sigma_arcsec=0.4, snr_min=10.0, omega_f=0.8, omega_v=0.04, c0=1.0)


@pytest.fixture(scope="module")
def field():
    fix = load("task_detection")
    img = np.tile(fix["stamp"], (NGAL, NGAL)).astype(np.float64)
    rng = np.random.default_rng(3)
    noise = rng.normal(scale=np.sqrt(VAR), size=img.shape)
    probe_noise = rng.normal(scale=np.sqrt(VAR), size=img.shape)
    cells = anacal.geometry.get_cell_list(
        img_nx=img.shape[1],
        img_ny=img.shape[0],
        cell_nx=512,
        cell_ny=512,
        cell_overlap=64,
        scale=SCALE,
    )
    return dict(
        img=img + noise, psf=fix["psf"], noise=probe_noise, cell=cells[0]
    )


def run(f, img=None, noise=None):
    cat = anacal.probe.process_cell(
        f["img"] if img is None else img,
        f["psf"],
        f["noise"] if noise is None else noise,
        VAR,
        KW["sigma_arcsec"],
        KW["snr_min"],
        KW["omega_f"],
        KW["omega_v"],
        KW["c0"],
        f["cell"],
    )
    order = np.lexsort((cat["x"], cat["y"]))
    return cat[order]


def slots(cat, stem, names):
    return np.stack(
        [cat[stem if n == "v" else f"{stem}_{n}"] for n in names], axis=1
    )


def test_matches_qnumber_detection(field):
    """
    Without renoising the qnumber detector and the probe see the same
    image, so the detections and the first-order slots of the weight and
    the background agree exactly (the same code, instantiated for the two
    number types).
    """
    cat = run(field)
    assert len(cat) == NGAL * NGAL
    qcat = anacal.detector.find_peaks(
        field["img"],
        field["psf"],
        KW["sigma_arcsec"],
        KW["snr_min"],
        VAR,
        KW["omega_f"],
        KW["omega_v"],
        field["cell"],
    )
    assert len(qcat) == len(cat)
    qx = np.array([round(g.x1_det / SCALE) for g in qcat])
    qy = np.array([round(g.x2_det / SCALE) for g in qcat])
    order = np.lexsort((qx, qy))
    np.testing.assert_array_equal(qx[order], cat["x"])
    np.testing.assert_array_equal(qy[order], cat["y"])

    cell = field["cell"]
    first = ["v", "g1", "g2", "x1", "x2"]
    for stem in ("wsel", "bkg"):
        q = []
        for k in order:
            g = qcat[k]
            a = getattr(g, stem).to_array()
            # find_peaks leaves the catalog anchored at the cell center;
            # the probe decentralizes to the detected pixel
            dx1 = -(g.x1_det - cell.xcen * SCALE)
            dx2 = -(g.x2_det - cell.ycen * SCALE)
            a[1] += dx1 * a[3] - dx2 * a[4]
            a[2] += dx1 * a[4] + dx2 * a[3]
            q.append(a)
        q = np.array(q)
        np.testing.assert_allclose(
            slots(cat, stem, first), q, rtol=1e-13, atol=1e-13 * np.abs(q).max()
        )


def test_noise_slots_are_derivatives(field):
    """
    The slots along the noise image against finite differences of the
    measurement on ``image + eps * noise``: the first-order ``u`` slot,
    the second-order ``u_u`` pair (as the derivative of the exact ``u``
    slot), and the mixed directions on a linear quantity (the moments'
    ``ug1, ug2, ux1``; for the ellipticity the derivative of its ``g1``
    slot also holds the pair ``(u, g1)``, which the layout does not
    carry).  The step is set by the float32 pixels.
    """
    cat = run(field)
    h = 0.02
    cp = run(field, img=field["img"] + h * field["noise"])
    cm = run(field, img=field["img"] - h * field["noise"])
    np.testing.assert_array_equal(cp["x"], cat["x"])
    np.testing.assert_array_equal(cm["x"], cat["x"])

    def fd(col):
        return (cp[col] - cm[col]) / (2 * h)

    for stem in ("S1", "S2", "e1", "e2", "m0", "m2", "bkg"):
        scale = max(np.abs(cat[f"{stem}_u"]).max(), 1e-12)
        np.testing.assert_allclose(
            fd(stem), cat[f"{stem}_u"], atol=5e-4 * scale, rtol=0
        )
        scale2 = max(np.abs(cat[f"{stem}_u_u"]).max(), 1e-12)
        np.testing.assert_allclose(
            fd(f"{stem}_u"), cat[f"{stem}_u_u"], atol=5e-4 * scale2, rtol=0
        )
    for stem in ("m0", "m2"):
        for d, ud in (
            ("g1", "ug1"),
            ("g2", "ug2"),
            ("x1", "ux1"),
            ("x2", "ux2"),
        ):
            scale = np.abs(cat[f"{stem}_{ud}"]).max()
            np.testing.assert_allclose(
                fd(f"{stem}_{d}"),
                cat[f"{stem}_{ud}"],
                atol=5e-4 * scale,
                rtol=0,
            )


def test_pair_slot_of_the_ellipticity(field):
    """
    The pair ``(u, ug1)`` of ``e1`` against finite differences, through
    the moments: ``e1 = m22 ffac2 / (m0 + c0)`` with ``m0`` and ``m22``
    linear in the image, so d/d eps of ``e1_g1`` is the ``ug1`` slot plus
    the second derivatives in the two moments times their ``u`` and
    ``g1`` slots; the same with ``e1_u`` and the ``ug1`` slots gives
    ``e1_u_ug1``.
    """
    cat = run(field)
    h = 0.02
    cp = run(field, img=field["img"] + h * field["noise"])
    cm = run(field, img=field["img"] - h * field["noise"])

    def fd(col):
        return (cp[col] - cm[col]) / (2 * h)

    sigma2 = KW["sigma_arcsec"] ** 2
    ffac = 0.5 / sigma2 * 0.318309886
    ffac2 = ffac * np.sqrt(2.0) / sigma2
    D = cat["m0"] + KW["c0"]
    m22 = cat["e1"] * D / ffac2

    def m22_slot(s):
        # m22 is linear: its slot from the product rule on e1 (m0 + c0)
        return (cat[f"e1_{s}"] * D + cat["e1"] * cat[f"m0_{s}"]) / ffac2

    # second derivatives of e1 in (m0, m22)
    d2_00 = 2.0 * m22 * ffac2 / D**3
    d2_02 = -ffac2 / D**2
    for d, ud in (("g1", "ug1"), ("g2", "ug2")):
        pair = d2_00 * cat["m0_u"] * cat[f"m0_{d}"] + d2_02 * (
            cat["m0_u"] * m22_slot(d) + m22_slot("u") * cat[f"m0_{d}"]
        )
        want = cat[f"e1_{ud}"] + pair
        scale = np.abs(want).max()
        np.testing.assert_allclose(
            fd(f"e1_{d}"), want, atol=5e-4 * scale, rtol=0
        )
        # and the pair slot itself: d/d eps of e1_ug1 is the pair (u, ug1)
        want2 = d2_00 * cat["m0_u"] * cat[f"m0_{ud}"] + d2_02 * (
            cat["m0_u"] * m22_slot(ud) + m22_slot("u") * cat[f"m0_{ud}"]
        )
        scale2 = max(np.abs(want2).max(), 1e-12)
        np.testing.assert_allclose(
            fd(f"e1_{ud}"), want2, atol=5e-4 * scale2, rtol=0
        )
        np.testing.assert_allclose(
            cat[f"e1_u_{ud}"], want2, atol=5e-4 * scale2, rtol=0
        )


def test_shear_estimate(field):
    """
    The stamp carries g1 = 0.03: the estimator recovers it within the
    noise of one field, with the noise-bias terms small against the
    response at this signal to noise.
    """
    cat = run(field)
    g1, g2 = anacal.probe.shear(cat)
    s = anacal.probe.sums(cat)
    assert abs(g1 - 0.03) < 0.005
    assert abs(g2) < 0.005
    assert abs(s["B1"] / s["R1"]) < 0.02
    assert np.isfinite(cat["S1_u_ug1"]).all()
    g1u, _ = anacal.probe.shear(cat, correct=False)
    assert abs(g1u - g1) < 0.002


def test_prepare_planes(field):
    """The planes: 17 slots, the probe's directions filled, pairs zero."""
    pl = anacal.probe.prepare_planes(
        field["img"],
        field["psf"],
        field["noise"],
        KW["sigma_arcsec"],
        field["cell"],
    )
    names = anacal.math.hnumber.names
    assert pl.shape == (len(names), field["cell"].ny, field["cell"].nx)
    assert np.isfinite(pl).all()
    # the rotated noise has the noise's variance
    u, ur = pl[names.index("u")], pl[names.index("ur")]
    np.testing.assert_allclose(ur.std(), u.std(), rtol=0.05)
    assert np.all(pl[11:] == 0.0)
    # the image's five planes equal the qnumber planes without renoising
    img_obj = anacal.image.ImageQ(
        nx=field["cell"].nx,
        ny=field["cell"].ny,
        scale=SCALE,
        sigma_arcsec=KW["sigma_arcsec"],
        klim=3.1 / SCALE,
    )
    q = img_obj.prepare_qnumber_image(
        field["img"], field["psf"], field["cell"].xcen, field["cell"].ycen
    )
    np.testing.assert_allclose(pl[:5], q, rtol=1e-12, atol=1e-12)


def test_slot_helpers_match_hnumber():
    """
    The vectorized product, chain rule and smooth step against the C++
    ``hnumber`` row by row.
    """
    from anacal import probe

    H = anacal.math.hnumber
    rng = np.random.default_rng(7)
    n = 50
    a = rng.normal(size=(n, probe.NSLOT))
    b = rng.normal(size=(n, probe.NSLOT))
    prod = probe.product(a, b)
    want = np.array([(H(a[i]) * H(b[i])).to_array() for i in range(n)])
    np.testing.assert_allclose(prod, want, rtol=1e-13, atol=1e-13)
    # the chain rule, through exp
    a[:, 0] = rng.uniform(0.5, 1.5, size=n)
    e = np.exp(a[:, 0])
    got = probe.chain(a, e, e, e)
    want = np.array([anacal.math.exp(H(a[i])).to_array() for i in range(n)])
    np.testing.assert_allclose(got, want, rtol=1e-13)
    # the smooth step, through the polynomial in hnumber arithmetic
    mu, sigma = 1.0, 0.6
    step = probe.ssfunc2(a, mu, sigma)
    for i in range(n):
        t = (a[i, 0] - mu) / (2 * sigma) + 0.5
        if t <= 0.0:
            want = np.zeros(probe.NSLOT)
        elif t >= 1.0:
            want = np.zeros(probe.NSLOT)
            want[0] = 1.0
        else:
            ht = (H(a[i]) - mu) / (2 * sigma) + 0.5
            pw = anacal.math.pow
            want = (
                6.0 * pw(ht, 5.0) - 15.0 * pw(ht, 4.0) + 10.0 * pw(ht, 3.0)
            ).to_array()
        np.testing.assert_allclose(step[i], want, rtol=1e-12, atol=1e-14)


def test_select(field):
    """
    A selection on the flux: a cut everything passes leaves the sums
    alone; a cut on the ramp reweights them, with weights in [0, 1] and
    the slots of the product rule; the shear estimate stays finite.
    """
    from anacal import probe

    cat = run(field)
    low = probe.select(cat, "m0", -1e6, 1.0)
    for stem in ("S1", "S2"):
        np.testing.assert_allclose(
            probe.slots(low, stem), probe.slots(cat, stem), rtol=1e-13
        )
    m0 = cat["m0"]
    mu = np.median(m0)
    sel = probe.select(cat, "m0", mu, 0.5 * mu)
    w = probe.ssfunc2(probe.slots(cat, "m0"), mu, 0.5 * mu)
    assert np.all(w[:, 0] >= 0.0) and np.all(w[:, 0] <= 1.0)
    assert 0.0 < w[:, 0].mean() < 1.0
    np.testing.assert_allclose(sel["S1"], w[:, 0] * cat["S1"], rtol=1e-13)
    # the response of the selected sum has the weight's response too
    np.testing.assert_allclose(
        sel["S1_g1"], w[:, 0] * cat["S1_g1"] + w[:, 1] * cat["S1"], rtol=1e-12
    )
    g1, g2 = probe.shear(sel)
    assert np.isfinite(g1) and np.isfinite(g2)
    assert abs(g1 - 0.03) < 0.01
