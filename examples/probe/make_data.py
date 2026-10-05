"""
Render the data of the probe example with GalSim.

A field of exponential galaxies at random positions, sheared by
g1 = +0.02 and, with the same galaxies and the same noise, by
g1 = -0.02, so that the two images pair; a Moffat PSF; and a pure-noise
realization with the covariance of the images' noise, which is what the
probe reads instead of adding noise to the image.  The shear is applied
to the shapes only: a lensed sky also moves the galaxies, but with the
positions held the two images share their noise under every galaxy and
the pair cancels the shape noise, which is what lets one small field
show the shear.  Everything a user would have in hand goes into one FITS
file, ``probe_data.fits``:

    image_plus, image_minus   the two images (float32)
    psf                       the PSF stamp, centered on (npsf // 2, npsf // 2)
    noise                     the noise realization, same shape as the images
    truth                     the galaxies' positions, fluxes and sizes

The header of ``image_plus`` carries the pixel scale (SCALE), the noise
variance (VAR) and the shear (G1, G2).  Run once:

    python make_data.py
"""

import galsim
import numpy as np
import fitsio

SEED = 11
SCALE = 0.2  # arcsec per pixel
DIM = 640  # pixels on a side
NGAL = 500
NOISE_STD = 0.4  # per pixel; a mag-30 zero point, LSST-like depth
PSF_FWHM = 0.8  # arcsec, Moffat beta 2.5 with a small anisotropy
PSF_G = (0.02, -0.01)
G1 = 0.02
NPSF = 51


def galaxy_population(rng, n):
    """Half-light radii, fluxes and intrinsic shapes of ``n`` galaxies."""
    hlr = rng.uniform(0.15, 0.7, size=n)
    # fluxes such that the detection sits in the sample: S/N from ~5 to ~100
    flux = 10 ** rng.uniform(1.8, 3.2, size=n)
    e = np.clip(rng.normal(scale=0.2, size=(n, 2)), -0.7, 0.7)
    x = rng.uniform(20, DIM - 20, size=n)
    y = rng.uniform(20, DIM - 20, size=n)
    return hlr, flux, e, x, y


def render(psf, hlr, flux, e, x, y, g1):
    """The field sheared by ``g1``, drawn with the PSF."""
    img = galsim.ImageF(DIM, DIM, scale=SCALE)
    for h, f, (e1, e2), xx, yy in zip(hlr, flux, e, x, y):
        gal = galsim.Exponential(half_light_radius=h, flux=f)
        gal = gal.shear(e1=e1, e2=e2).shear(g1=g1, g2=0.0)
        obj = galsim.Convolve(gal, psf)
        stamp = obj.drawImage(scale=SCALE, center=galsim.PositionD(xx, yy))
        bounds = stamp.bounds & img.bounds
        if bounds.isDefined():
            img[bounds] += stamp[bounds]
    return img.array


def main():
    rng = np.random.default_rng(SEED)
    psf = galsim.Moffat(beta=2.5, fwhm=PSF_FWHM).shear(g1=PSF_G[0], g2=PSF_G[1])
    psf_arr = psf.drawImage(nx=NPSF, ny=NPSF, scale=SCALE).array
    hlr, flux, e, x, y = galaxy_population(rng, NGAL)

    noise = rng.normal(scale=NOISE_STD, size=(DIM, DIM)).astype(np.float32)
    probe_noise = rng.normal(scale=NOISE_STD, size=(DIM, DIM)).astype(
        np.float32
    )

    header = {"SCALE": SCALE, "VAR": NOISE_STD**2, "G2": 0.0}
    with fitsio.FITS("probe_data.fits", "rw", clobber=True) as fits:
        for name, g1 in (("image_plus", G1), ("image_minus", -G1)):
            img = render(psf, hlr, flux, e, x, y, g1) + noise
            fits.write(img, extname=name, header={**header, "G1": g1})
        fits.write(psf_arr, extname="psf", header={"SCALE": SCALE})
        fits.write(probe_noise, extname="noise", header={"VAR": NOISE_STD**2})
        truth = np.zeros(
            NGAL,
            dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8"), ("hlr", "f8")],
        )
        truth["x"], truth["y"], truth["flux"], truth["hlr"] = x, y, flux, hlr
        fits.write(truth, extname="truth")
    print("wrote probe_data.fits")


if __name__ == "__main__":
    main()
