#include "anacal.h"

namespace anacal {
namespace probe {

void
pyExportProbe(py::module_& m) {
    py::module_ probe = m.def_submodule(
        "probe",
        "Detection and moments with the noise-bias probe (math.hnumber)."
    );
    probe.def(
        "process_cell",
        [](
            const py::array_t<pixel_t>& img_array,
            const py::array_t<double>& psf_array,
            const py::array_t<pixel_t>& noise_array,
            double variance,
            double sigma_arcsec,
            double snr_min,
            double omega_f,
            double omega_v,
            double c0,
            const geometry::cell & cell,
            int image_bound
        ) {
            std::vector<probeDetection> cat = process_cell_impl(
                img_array, psf_array, noise_array, variance,
                sigma_arcsec, snr_min, omega_f, omega_v, c0, cell,
                image_bound
            );
            return to_array(cat, cell.scale);
        },
        "Detect and measure one cell with the noise-bias probe.\n"
        "\n"
        "The detection of ``detector.find_peaks`` and the Gaussian-weighted\n"
        "moments of the fitter's ``fpfs_*`` columns, taken about the\n"
        "detected pixel, over ``math.hnumber`` pixels: every quantity\n"
        "carries, besides the image's shear and translation responses, its\n"
        "responses along the noise image and the second-order slots of the\n"
        "noise-bias correction.  The noise is not added to the image.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "img_array : ndarray, (ny, nx)\n"
        "    The image, one band.\n"
        "psf_array : ndarray\n"
        "    The PSF stamp, centered on pixel (npsf // 2, npsf // 2).\n"
        "noise_array : ndarray, (ny, nx)\n"
        "    A pure-noise realization with the covariance of the image's\n"
        "    noise.\n"
        "variance : float\n"
        "    The pixel variance of the image's noise.\n"
        "sigma_arcsec : float\n"
        "    The measurement scale; detection smooths at sqrt(2) times it.\n"
        "snr_min : float\n"
        "    The flux gate of the detection, in units of the smoothed noise.\n"
        "omega_f, omega_v : float\n"
        "    The ramp widths of the flux gates and of the neighbor\n"
        "    differences, as in ``detector.find_peaks``.\n"
        "c0 : float\n"
        "    The constant of the ellipticity's denominator, in the units of\n"
        "    ``m0``.\n"
        "cell : geometry.cell\n"
        "    The cell.\n"
        "image_bound : int, optional\n"
        "    The width of the image border in which nothing is detected.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "ndarray\n"
        "    A structured array with ``x, y`` (the detected pixel),\n"
        "    ``cell_id``, and for each of ``wsel, bkg, m0, m2, e1, e2, S1,\n"
        "    S2`` one column per slot of ``math.hnumber``: the value under\n"
        "    the name, then ``<name>_<slot>`` in ``hnumber.names`` order.\n"
        "    ``S1 = wsel * e1``; the shear estimator from the array is\n"
        "    ``anacal.probe.shear``.",
        py::arg("img_array"),
        py::arg("psf_array"),
        py::arg("noise_array"),
        py::arg("variance"),
        py::arg("sigma_arcsec"),
        py::arg("snr_min"),
        py::arg("omega_f"),
        py::arg("omega_v"),
        py::arg("c0"),
        py::arg("cell"),
        py::arg("image_bound")=0
    );
    probe.def(
        "prepare_planes",
        &prepare_planes,
        "The probe's planes of a cell.\n"
        "\n"
        "The image through the Gaussian kernel at ``sigma_arcsec``, its\n"
        "shear and translation responses, the same five planes of the\n"
        "noise image, the noise through the rotated kernel, and the pairs,\n"
        "which are zero at a pixel; anchored at the cell center.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "img_array : ndarray, (ny, nx)\n"
        "    The image, one band.\n"
        "psf_array : ndarray\n"
        "    The PSF stamp.\n"
        "noise_array : ndarray, (ny, nx)\n"
        "    The pure-noise realization.\n"
        "sigma_arcsec : float\n"
        "    The smoothing scale.\n"
        "cell : geometry.cell\n"
        "    The cell.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "ndarray, (hnumber.NSLOT, ny, nx)\n"
        "    The planes, in ``hnumber.names`` order.",
        py::arg("img_array"),
        py::arg("psf_array"),
        py::arg("noise_array"),
        py::arg("sigma_arcsec"),
        py::arg("cell")
    );
    probe.attr("NSLOT") = py::int_(math::hnumber::NSLOT);
}

} // namespace probe
} // namespace anacal
