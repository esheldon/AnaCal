"""
Shear measurement with the noise-bias probe, from data on disk.

The scenario: a user has an image, its PSF, the noise variance and a
pure-noise realization with the covariance of the image's noise (the
noise coadd that a coadding pipeline produces alongside the coadd).
With AnaCal's probe the measurement is

1. detect and measure every cell with ``anacal.probe.process_cell``,
   which returns a catalog in which every quantity carries, besides its
   shear responses, its responses along the noise realization and the
   second-order slots of the noise-bias correction (``math.hnumber``);
2. concatenate the cells and write the catalog to a file -- all slots go
   with it, so the selection can be made later, by someone else;
3. read it back, select on the measured flux and size with smooth cuts
   (``anacal.probe.select``), which carries the cuts' own responses and
   noise-bias terms into the sums;
4. estimate the shear with ``anacal.probe.shear``.

The data here come from ``make_data.py``: the same field sheared by
g1 = +0.02 and by -0.02 with the same noise and the galaxies at the
same pixels, so the two estimates pair and ``(g_plus - g_minus) / 2``
recovers the shear with the shape noise cancelled.  The uncorrected
estimate (``correct=False``, the plain sum of shapes over the sum of
responses) is printed beside it to show the size of the noise-bias
correction; on one field of a few hundred galaxies the correction is
within the pair's own scatter, and it takes the ensemble tests of the
documentation to measure it.
"""

import os

import anacal
import fitsio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "probe_data.fits")
# the catalogs are written to the working directory
OUT = os.getcwd()

# the measurement: the detection and smoothing scale, the detection
# threshold and the ramps of the smooth gates (as anacal.task.Task), and
# the constant of the ellipticity's denominator
SIGMA_ARCSEC = 0.45
SNR_MIN = 10.0
OMEGA_F = 0.8
OMEGA_V = 0.04
C0 = 1.0

# the cells: a field is processed in cells with an overlap at least as
# wide as the detector's reach (see detector.h; 2 * 16 pixels at 0.2")
CELL = 256
OVERLAP = 64


def measure_field(image, psf, noise, variance, scale):
    """
    The probe catalog of a field: every cell, concatenated.

    Parameters
    ----------
    image, noise : ndarray, (ny, nx)
        The image and the noise realization.
    psf : ndarray
        The PSF stamp.
    variance : float
        The pixel variance of the noise.
    scale : float
        The pixel scale, arcsec.

    Returns
    -------
    ndarray
        The structured catalog of ``anacal.probe.process_cell`` over the
        field.
    """
    ny, nx = image.shape
    cells = anacal.geometry.get_cell_list(
        img_nx=nx,
        img_ny=ny,
        cell_nx=CELL,
        cell_ny=CELL,
        cell_overlap=OVERLAP,
        scale=scale,
    )
    cats = [
        anacal.probe.process_cell(
            image,
            psf,
            noise,
            variance,
            SIGMA_ARCSEC,
            SNR_MIN,
            OMEGA_F,
            OMEGA_V,
            C0,
            cell,
        )
        for cell in cells
    ]
    return np.concatenate(cats)


def main():
    # --- 1. the data in hand --------------------------------------------
    with fitsio.FITS(DATA) as fits:
        header = fits["image_plus"].read_header()
        scale = header["SCALE"]
        variance = header["VAR"]
        g1_true = header["G1"]
        images = {
            sign: fits[f"image_{sign}"].read().astype(np.float32)
            for sign in ("plus", "minus")
        }
        psf = fits["psf"].read().astype(np.float64)
        noise = fits["noise"].read().astype(np.float32)

    # --- 2. detect and measure, and write the catalogs -------------------
    for sign, image in images.items():
        cat = measure_field(image, psf, noise, variance, scale)
        fname = os.path.join(OUT, f"catalog_{sign}.fits")
        fitsio.write(fname, cat, clobber=True)
        print(f"{sign}: {len(cat)} detections -> {fname}")

    # --- 3. read back and select ------------------------------------------
    # The flux m0 is the Gaussian-weighted flux at the measurement scale;
    # its noise for a point source is the smoothed noise level, which the
    # catalog does not carry, so the S/N scale is set from the data: the
    # faintest detections sit at the detection threshold.  A smooth cut
    # at twice that, with a ramp of half its width, keeps the sample well
    # above the threshold; a second cut on the size m2 removes the
    # unresolved end.  Both cuts are differentiable, and ``select``
    # carries their responses and noise-bias terms into the sums.
    estimates = {}
    for sign in ("plus", "minus"):
        cat = fitsio.read(os.path.join(OUT, f"catalog_{sign}.fits"))
        flux_floor = np.percentile(cat["m0"], 5.0)
        cut_flux = 2.0 * flux_floor
        sel = anacal.probe.select(cat, "m0", cut_flux, 0.5 * cut_flux)
        size_floor = np.percentile(cat["m2"], 10.0)
        sel = anacal.probe.select(sel, "m2", size_floor, 0.5 * abs(size_floor))

        # the effective number kept: the selection weight is the ratio of
        # the selected to the unselected summands' weights
        w = anacal.probe.ssfunc2(
            anacal.probe.slots(cat, "m0"), cut_flux, 0.5 * cut_flux
        )[:, 0]
        print(
            f"{sign}: flux cut at {cut_flux:.1f}, size cut at "
            f"{size_floor:.3f}; {w.sum():.0f} of {len(cat)} kept by the "
            "flux ramp"
        )

        # --- 4. the shear --------------------------------------------------
        estimates[sign] = {
            "corrected": anacal.probe.shear(sel),
            "uncorrected": anacal.probe.shear(sel, correct=False),
            "no cuts": anacal.probe.shear(cat),
        }
        s = anacal.probe.sums(sel)
        print(
            f"{sign}: R1 = {s['R1']:.1f}, noise-bias term B1/R1 = "
            f"{s['B1'] / s['R1']:+.4f}, additive term C1/N1 = "
            f"{s['C1'] / s['N1']:+.4f}"
        )

    # the pair: the difference cancels the shape noise, and the sum of
    # the two (not printed) does not, so only the difference is a
    # measurement on one field
    print()
    print(f"true shear g1 = {g1_true:+.4f}")
    for key in ("corrected", "uncorrected", "no cuts"):
        gp = estimates["plus"][key]
        gm = estimates["minus"][key]
        g1 = 0.5 * (gp[0] - gm[0])
        print(
            f"{key:12s}: g1 = {g1:+.4f} (m = {g1 / g1_true - 1:+.3f}) "
            f"from g1(plus) = {gp[0]:+.4f}, g1(minus) = {gm[0]:+.4f}"
        )


if __name__ == "__main__":
    main()
