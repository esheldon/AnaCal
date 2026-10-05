#ifndef ANACAL_PROBE_H
#define ANACAL_PROBE_H

#include "detector.h"

#include <cstring>

// Detection and Gaussian-moment shapes with the noise-bias probe.
//
// The same detection as detector.h and the same moments as the fitter's
// ``measure_fpfs`` (ngmix/fitting.h), taken about the detected pixel, over
// math::hnumber pixels (math/dual.h): every quantity carries, besides the
// image's shear and translation responses, the responses along a
// pure-noise image and the second-order slots of the noise-bias correction.
// The catalog is written as a structured numpy array with one column per
// slot (``wsel, wsel_g1, ..., wsel_u_ug1, ...``), and the shear estimator
// from it is, over the rows,
//
//     g1 = sum(S1 - C1) / sum(R1 - B1),
//     R1 = S1_g1,  B1 = S1_u_ug1,  C1 = 1/4 (S1_u_u - S1_ur_ur),
//
// with S1 = wsel e1 (python/anacal/probe.py).  Nothing of the qnumber
// path is touched: this is a separate entry point beside it, and the
// detector's core is shared with it through its templates.

namespace anacal {
namespace probe {

// A detection with its moments, all slots carried.  The members the
// detector writes are those of table::galNumber that it uses (see
// detector::measure_pixel); the rest are set by measure_moments.
struct probeDetection {
    double x1_det = 0.0;   // detected position, arcsec (pixel * scale)
    double x2_det = 0.0;
    int cell_id = -1;
    math::hnumber bkg;
    math::hnumber wsel;
    math::hnumber m0;      // fpfs_m0, the Gaussian-weighted flux
    math::hnumber m2;      // fpfs_m2
    math::hnumber e1;      // fpfs_e1 = (mxx - myy) ffac2 / (m0 + c0)
    math::hnumber e2;
    math::hnumber S1;      // wsel e1, the estimator's summands
    math::hnumber S2;

    void shift_reference(double dx1, double dx2) {
        this->bkg.shift_reference(dx1, dx2);
        this->wsel.shift_reference(dx1, dx2);
        this->m0.shift_reference(dx1, dx2);
        this->m2.shift_reference(dx1, dx2);
        this->e1.shift_reference(dx1, dx2);
        this->e2.shift_reference(dx1, dx2);
        this->S1.shift_reference(dx1, dx2);
        this->S2.shift_reference(dx1, dx2);
    };

    // As galNumber: the planes are anchored at the cell center, the
    // catalog at the detected pixel.
    void decentralize(const geometry::cell & cell) {
        const double dx1 = this->x1_det - cell.xcen * cell.scale;
        const double dx2 = this->x2_det - cell.ycen * cell.scale;
        this->shift_reference(-dx1, -dx2);
    };
};

// The constants of the moments, as GaussFit sets them (ngmix/fitting.h),
// so that the columns mean the same as the fitter's fpfs_* columns.
struct momentSetup {
    double sigma2;
    double rfac;
    double ffac;
    double ffac2;
    double ffac3;
    double sigma2_lim;
    double c0;
    int r_pix;

    momentSetup(double sigma_arcsec, double scale, double c0) : c0(c0) {
        constexpr double sqrt2 = 1.4142135623730951;
        this->sigma2 = sigma_arcsec * sigma_arcsec;
        const double sigma_m2 = 1.0 / this->sigma2;
        this->rfac = -0.5 * sigma_m2;
        this->ffac = this->rfac * (-0.318309886);
        this->ffac2 = this->ffac * sqrt2 * sigma_m2;
        this->ffac3 = this->ffac * 2.0 * sigma_m2;
        this->sigma2_lim = this->sigma2 * 20;
        this->r_pix = static_cast<int>(sigma_arcsec * 8 / scale);
    };
};

// The Gaussian-weighted moments about the detected pixel.  The weight is a
// fixed function of the offset from that pixel, so the moments are linear
// in the planes and the ellipticity's slots come from the division alone.
inline void
measure_moments(
    const std::vector<math::hnumber> & data,
    probeDetection & src,
    const geometry::cell & cell,
    const momentSetup & ms
) {
    const int i_cen = static_cast<int>(std::round(src.x1_det / cell.scale))
        - cell.xmin;
    const int j_cen = static_cast<int>(std::round(src.x2_det / cell.scale))
        - cell.ymin;
    const int i_min = std::max(i_cen - ms.r_pix, 0);
    const int i_max = std::min(i_cen + ms.r_pix + 1, cell.nx);
    const int j_min = std::max(j_cen - ms.r_pix, 0);
    const int j_max = std::min(j_cen + ms.r_pix + 1, cell.ny);

    math::hnumber m0, mxx, myy, mxy;
    for (int j = j_min; j < j_max; ++j) {
        if (!cell.ymsk[j]) continue;
        const int jj = j * cell.nx;
        const double ys = cell.yvs[j] - src.x2_det;
        const double y2 = ys * ys;
        for (int i = i_min; i < i_max; ++i) {
            if (!cell.xmsk[i]) continue;
            const double xs = cell.xvs[i] - src.x1_det;
            const double x2 = xs * xs;
            if ((x2 + y2) < ms.sigma2_lim) {
                const double w = std::exp((x2 + y2) * ms.rfac);
                const math::hnumber wf = data[jj + i] * w;
                m0 = m0 + wf;
                mxx = mxx + wf * x2;
                myy = myy + wf * y2;
                mxy = mxy + wf * (xs * ys);
            }
        }
    }
    src.m0 = m0 * ms.ffac;
    src.m2 = (mxx + myy - m0 * ms.sigma2) * ms.ffac3;
    const math::hnumber denom = src.m0 + ms.c0;
    src.e1 = (mxx - myy) * ms.ffac2 / denom;
    src.e2 = mxy * (2.0 * ms.ffac2) / denom;
    src.S1 = src.wsel * src.e1;
    src.S2 = src.wsel * src.e2;
};

// Detect and measure one cell with the probe.  Single band: the image,
// the PSF and the noise image are 2-D arrays and ``variance`` is the
// pixel variance.  The planes are built twice, at the detection scale
// (sqrt(2) sigma_arcsec, as detector.h) and at the measurement scale.
inline std::vector<probeDetection>
process_cell_impl(
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
    if (img_array.ndim() != 2) {
        throw std::runtime_error(
            "probe Error: the probe takes a single band (2-D arrays)"
        );
    }
    const double sigma_det = detection_sigma(sigma_arcsec);
    const std::vector<double> variance_vec{variance};
    const std::vector<double> w{1.0};
    const double std_noise = std::sqrt(
        coadd_smoothed_variance(
            cell.scale, sigma_det, psf_array, variance_vec, w
        )
    );

    std::vector<probeDetection> cat;
    {
        const std::vector<math::hnumber> data = prepare_data_cell_probe(
            img_array, psf_array, noise_array, sigma_det, cell
        );
        cat = detector::find_peaks_from_data_t<probeDetection>(
            data,
            std_noise,
            snr_min,
            omega_f,
            omega_v,
            cell,
            static_cast<int>(img_array.shape(0)),
            static_cast<int>(img_array.shape(1)),
            image_bound
        );
    }
    if (cat.empty()) return cat;

    const std::vector<math::hnumber> data = prepare_data_cell_probe(
        img_array, psf_array, noise_array, sigma_arcsec, cell
    );
    const momentSetup ms(sigma_arcsec, cell.scale, c0);
    for (probeDetection& src : cat) {
        measure_moments(data, src, cell, ms);
        src.decentralize(cell);
    }
    return cat;
};

// The catalog as a structured array: ``x, y`` (the detected pixel),
// ``cell_id``, then for each quantity one column per slot, named
// ``<quantity>`` for the value and ``<quantity>_<slot>`` for the others,
// in math::ProbeLayout::names order.
inline py::array
to_array(const std::vector<probeDetection>& cat, double scale) {
    using L = math::ProbeLayout;
    constexpr int nslot = math::hnumber::NSLOT;
    static const char* quantities[] = {
        "wsel", "bkg", "m0", "m2", "e1", "e2", "S1", "S2"
    };
    constexpr int nq = 8;

    std::vector<std::string> names;
    std::vector<std::string> formats;
    std::vector<ssize_t> offsets;
    ssize_t offset = 0;
    auto add = [&](const std::string& name, const char* fmt, ssize_t size) {
        names.push_back(name);
        formats.push_back(fmt);
        offsets.push_back(offset);
        offset += size;
    };
    add("x", "<i8", 8);
    add("y", "<i8", 8);
    add("cell_id", "<i8", 8);
    for (int q = 0; q < nq; ++q) {
        add(quantities[q], "<f8", 8);
        for (int k = 1; k < nslot; ++k) {
            add(std::string(quantities[q]) + "_" + L::names[k], "<f8", 8);
        }
    }
    const ssize_t itemsize = offset;
    py::list py_names, py_formats, py_offsets;
    for (std::size_t k = 0; k < names.size(); ++k) {
        py_names.append(py::str(names[k]));
        py_formats.append(py::str(formats[k]));
        py_offsets.append(py::int_(offsets[k]));
    }
    py::dtype dt(py_names, py_formats, py_offsets, itemsize);
    // the shape as a vector: a braced list here would pick the
    // array(handle, bool) constructor
    const std::vector<ssize_t> shape{static_cast<ssize_t>(cat.size())};
    py::array out(dt, shape);

    char* base = static_cast<char*>(out.mutable_data());
    for (std::size_t n = 0; n < cat.size(); ++n) {
        const probeDetection& src = cat[n];
        char* row = base + n * itemsize;
        const math::hnumber* qs[nq] = {
            &src.wsel, &src.bkg, &src.m0, &src.m2,
            &src.e1, &src.e2, &src.S1, &src.S2
        };
        auto put_i = [&](int col, int64_t val) {
            std::memcpy(row + offsets[col], &val, sizeof(val));
        };
        put_i(0, static_cast<int64_t>(std::lround(src.x1_det / scale)));
        put_i(1, static_cast<int64_t>(std::lround(src.x2_det / scale)));
        put_i(2, static_cast<int64_t>(src.cell_id));
        int col = 3;
        for (int q = 0; q < nq; ++q) {
            const std::array<double, nslot> s = qs[q]->slots();
            for (int k = 0; k < nslot; ++k, ++col) {
                std::memcpy(row + offsets[col], &s[k], sizeof(double));
            }
        }
    }
    return out;
};

// The probe's planes of a cell as a (NSLOT, ny, nx) array, for tests.
inline py::array_t<double>
prepare_planes(
    const py::array_t<pixel_t>& img_array,
    const py::array_t<double>& psf_array,
    const py::array_t<pixel_t>& noise_array,
    double sigma_arcsec,
    const geometry::cell & cell
) {
    constexpr int nslot = math::hnumber::NSLOT;
    const std::vector<math::hnumber> data = prepare_data_cell_probe(
        img_array, psf_array, noise_array, sigma_arcsec, cell
    );
    auto result = py::array_t<double>({nslot, cell.ny, cell.nx});
    auto r = result.mutable_unchecked<3>();
    for (ssize_t j = 0, idx = 0; j < cell.ny; ++j) {
        for (ssize_t i = 0; i < cell.nx; ++i, ++idx) {
            const std::array<double, nslot> s = data[idx].slots();
            for (int k = 0; k < nslot; ++k) {
                r(k, j, i) = s[k];
            }
        }
    }
    return result;
};

void pyExportProbe(py::module_& m);

} // namespace probe
} // namespace anacal

#endif // ANACAL_PROBE_H
