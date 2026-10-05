#ifndef ANACAL_MATH_DUAL_H
#define ANACAL_MATH_DUAL_H

#include "../stdafx.h"

// A multi-dual number generic in its slot layout.
//
// ``dual<L>`` carries a value, one first-order slot per DIRECTION of the
// layout ``L`` and one second-order slot per PAIR of directions that ``L``
// lists.  The arithmetic is that of hyper-dual numbers truncated to the
// slots the layout carries (a pair not in the table is not propagated):
//
//     (a b).v    = a.v b.v
//     (a b).d    = a.d b.v + a.v b.d                (each direction d)
//     (a b).pq   = a.pq b.v + a.v b.pq + a.p b.q + a.q b.p
//     f(a).d     = f'(a.v) a.d
//     f(a).pq    = f'(a.v) a.pq + f''(a.v) a.p a.q
//
// so a slot is an exact derivative of the result along its direction(s),
// however many operations deep.  Division is a product with the
// reciprocal (f' = -1/x^2, f'' = 2/x^3).  The pair formula is written for
// p == q too, where it reads 2 a.p b.p.
//
// A layout is a struct of compile-time tables (see ``ProbeLayout`` below):
// the directions as an enum, the pairs, and the SHEAR table that says how
// the shear slots move when the reference point of the coordinates moves
// by (dx1, dx2), which is what ``qnumber::shift_reference`` does for its
// g1/g2 from x1/x2.  The loops over those tables unroll at -O2, so an
// instance costs what the same slots written out by hand would.
//
// The layout that motivates this is the noise-bias PROBE: besides the
// image's shear and translation responses (as qnumber), the directions
// u, ug1, ug2, ux1, ux2 are the same five planes of a pure-noise image
// read at the pixel, ur is the 90-degree-rotated noise, and the pairs
// (u, ug1), (u, ug2), (u, u), (ur, ur) plus the re-anchoring pairs (u, ux1),
// (u, ux2) hold the second derivatives that the noise-bias correction of a
// response-calibrated estimator needs (Stein's identity, exact for
// Gaussian noise).  For an estimator S = w e summed over detections,
//
//     g = sum(S - C) / sum(R - B),
//     R = sum S_g1,  B = sum S_u_ug1,  C = 1/4 sum (S_u_u - S_ur_ur),
//
// with no noise added to the image (unlike renoising) and no model of
// the noise covariance: the probe is a realization with the covariance
// of the data's noise.

namespace anacal {
namespace math {

// One entry of a layout's shear table: when the reference point moves by
// dx = (dx1, dx2), slot ``shear`` gains ``sign * dx[comp] * slot(trans)``.
struct shift_entry {
    int shear;
    int trans;
    int sign;
    int comp;
};

// The layout of the noise-bias probe.
struct ProbeLayout {
    enum dir { g1, g2, x1, x2, u, ug1, ug2, ux1, ux2, ur };
    static constexpr int NDIR = 10;
    static constexpr int NPAIR = 6;
    static constexpr std::array<std::pair<int, int>, NPAIR> pairs{{
        {u, ug1}, {u, ug2}, {u, ux1}, {u, ux2}, {u, u}, {ur, ur}
    }};
    // AnaCal's convention (qnumber.h): g1 += dx1 x1 - dx2 x2,
    // g2 += dx1 x2 + dx2 x1; the probe's tangents move the same way.
    static constexpr std::array<shift_entry, 8> shear{{
        {g1, x1, +1, 0}, {g1, x2, -1, 1},
        {g2, x2, +1, 0}, {g2, x1, +1, 1},
        {ug1, ux1, +1, 0}, {ug1, ux2, -1, 1},
        {ug2, ux2, +1, 0}, {ug2, ux1, +1, 1}
    }};
    // field names, value first, then the directions, then the pairs
    static constexpr std::array<const char*, 1 + NDIR + NPAIR> names{{
        "v",
        "g1", "g2", "x1", "x2", "u", "ug1", "ug2", "ux1", "ux2", "ur",
        "u_ug1", "u_ug2", "u_ux1", "u_ux2", "u_u", "ur_ur"
    }};
};

namespace detail {

// The pair slot of (p, q) in either order, or -1.
template <class L>
constexpr int pair_index(int p, int q) {
    for (int k = 0; k < L::NPAIR; ++k) {
        if ((L::pairs[k].first == p && L::pairs[k].second == q) ||
            (L::pairs[k].first == q && L::pairs[k].second == p)) {
            return k;
        }
    }
    return -1;
}

// Every pair (p, g) with g a shear direction must have its partner
// (p, x) in the table for each translation direction x of g, so that the
// pair can be re-anchored.  Checked at compile time.
template <class L>
constexpr bool shear_table_closed() {
    for (int k = 0; k < L::NPAIR; ++k) {
        const int p = L::pairs[k].first;
        const int q = L::pairs[k].second;
        for (const shift_entry& e : L::shear) {
            if (e.shear == q && pair_index<L>(p, e.trans) < 0) return false;
            if (e.shear == p && pair_index<L>(q, e.trans) < 0) return false;
        }
    }
    return true;
}

} // namespace detail

template <class L>
struct dual {
    static constexpr int NDIR = L::NDIR;
    static constexpr int NPAIR = L::NPAIR;
    static constexpr int NSLOT = 1 + NDIR + NPAIR;
    using layout = L;

    static_assert(
        detail::shear_table_closed<L>(),
        "a pair with a shear direction needs its translation partners"
    );

    double v = 0.0;
    std::array<double, NDIR> d{};
    std::array<double, NPAIR> p{};

    dual() = default;

    explicit dual(double v) : v(v) {}

    // from the flat slots, value first, in ``L::names`` order
    explicit dual(const std::array<double, NSLOT>& a) {
        this->v = a[0];
        for (int k = 0; k < NDIR; ++k) this->d[k] = a[1 + k];
        for (int k = 0; k < NPAIR; ++k) this->p[k] = a[1 + NDIR + k];
    }

    std::array<double, NSLOT> slots() const {
        std::array<double, NSLOT> a;
        a[0] = this->v;
        for (int k = 0; k < NDIR; ++k) a[1 + k] = this->d[k];
        for (int k = 0; k < NPAIR; ++k) a[1 + NDIR + k] = this->p[k];
        return a;
    }

    // a slot by its index in ``L::names`` order
    double slot(int k) const {
        if (k == 0) return this->v;
        if (k <= NDIR) return this->d[k - 1];
        return this->p[k - 1 - NDIR];
    }

    // --- arithmetic -------------------------------------------------------

    dual operator-() const {
        dual r;
        r.v = -this->v;
        for (int k = 0; k < NDIR; ++k) r.d[k] = -this->d[k];
        for (int k = 0; k < NPAIR; ++k) r.p[k] = -this->p[k];
        return r;
    }

    dual operator+(const dual& o) const {
        dual r;
        r.v = this->v + o.v;
        for (int k = 0; k < NDIR; ++k) r.d[k] = this->d[k] + o.d[k];
        for (int k = 0; k < NPAIR; ++k) r.p[k] = this->p[k] + o.p[k];
        return r;
    }

    dual operator-(const dual& o) const {
        dual r;
        r.v = this->v - o.v;
        for (int k = 0; k < NDIR; ++k) r.d[k] = this->d[k] - o.d[k];
        for (int k = 0; k < NPAIR; ++k) r.p[k] = this->p[k] - o.p[k];
        return r;
    }

    dual operator*(const dual& o) const {
        dual r;
        r.v = this->v * o.v;
        for (int k = 0; k < NDIR; ++k) {
            r.d[k] = this->d[k] * o.v + this->v * o.d[k];
        }
        for (int k = 0; k < NPAIR; ++k) {
            const int i = L::pairs[k].first;
            const int j = L::pairs[k].second;
            r.p[k] = this->p[k] * o.v + this->v * o.p[k]
                   + this->d[i] * o.d[j] + this->d[j] * o.d[i];
        }
        return r;
    }

    dual operator/(const dual& o) const {
        return *this * recip(o);
    }

    friend dual operator+(const dual& a, double b) {
        dual r = a;
        r.v += b;
        return r;
    }
    friend dual operator+(double a, const dual& b) { return b + a; }

    friend dual operator-(const dual& a, double b) {
        dual r = a;
        r.v -= b;
        return r;
    }
    friend dual operator-(double a, const dual& b) {
        dual r = -b;
        r.v += a;
        return r;
    }

    friend dual operator*(const dual& a, double b) {
        dual r;
        r.v = a.v * b;
        for (int k = 0; k < NDIR; ++k) r.d[k] = a.d[k] * b;
        for (int k = 0; k < NPAIR; ++k) r.p[k] = a.p[k] * b;
        return r;
    }
    friend dual operator*(double a, const dual& b) { return b * a; }

    friend dual operator/(const dual& a, double b) { return a * (1.0 / b); }
    friend dual operator/(double a, const dual& b) { return recip(b) * a; }

    // --- the chain rule ---------------------------------------------------

    // f(a) from f, f' and f'' at a.v
    static dual apply(const dual& a, double f0, double f1, double f2) {
        dual r;
        r.v = f0;
        for (int k = 0; k < NDIR; ++k) r.d[k] = f1 * a.d[k];
        for (int k = 0; k < NPAIR; ++k) {
            const int i = L::pairs[k].first;
            const int j = L::pairs[k].second;
            r.p[k] = f1 * a.p[k] + f2 * a.d[i] * a.d[j];
        }
        return r;
    }

    static dual recip(const dual& a) {
        const double inv = 1.0 / a.v;
        return apply(a, inv, -inv * inv, 2.0 * inv * inv * inv);
    }

    // --- re-anchoring -----------------------------------------------------

    // In place, as qnumber::shift_reference: the shear slots gain the
    // position-coupling terms of a move of the reference point by
    // (dx1, dx2), and so do the pairs that hold a shear direction, through
    // the pairs that hold its translation partner.  The slots written are
    // never read, so the order is alias-free.
    void shift_reference(double dx1, double dx2) {
        const double dx[2] = {dx1, dx2};
        for (const shift_entry& e : L::shear) {
            this->d[e.shear] += e.sign * dx[e.comp] * this->d[e.trans];
        }
        for (int k = 0; k < NPAIR; ++k) {
            const int i = L::pairs[k].first;
            const int j = L::pairs[k].second;
            double add = 0.0;
            for (const shift_entry& e : L::shear) {
                if (e.shear == j) {
                    add += e.sign * dx[e.comp]
                         * this->p[detail::pair_index<L>(i, e.trans)];
                }
                if (e.shear == i) {
                    add += e.sign * dx[e.comp]
                         * this->p[detail::pair_index<L>(j, e.trans)];
                }
            }
            this->p[k] += add;
        }
    }

    void centralize(double dx1, double dx2) {
        this->shift_reference(dx1, dx2);
    }

    void decentralize(double dx1, double dx2) {
        this->shift_reference(-dx1, -dx2);
    }

    friend std::ostream& operator<<(std::ostream& os, const dual& q) {
        os << L::names[0] << ": " << q.v;
        for (int k = 1; k < NSLOT; ++k) {
            os << ", " << L::names[k] << ": " << q.slot(k);
        }
        return os;
    }
};

// --- functions of a dual, by the chain rule ----------------------------------

template <class L>
inline dual<L> exp(const dual<L>& a) {
    const double e = std::exp(a.v);
    return dual<L>::apply(a, e, e, e);
}

template <class L>
inline dual<L> log(const dual<L>& a) {
    const double inv = 1.0 / a.v;
    return dual<L>::apply(a, std::log(a.v), inv, -inv * inv);
}

template <class L>
inline dual<L> sqrt(const dual<L>& a) {
    const double s = std::sqrt(a.v);
    return dual<L>::apply(a, s, 0.5 / s, -0.25 / (a.v * s));
}

template <class L>
inline dual<L> tanh(const dual<L>& a) {
    const double t = std::tanh(a.v);
    const double dt = 1.0 - t * t;
    return dual<L>::apply(a, t, dt, -2.0 * t * dt);
}

template <class L>
inline dual<L> sin(const dual<L>& a) {
    const double s = std::sin(a.v);
    const double c = std::cos(a.v);
    return dual<L>::apply(a, s, c, -s);
}

template <class L>
inline dual<L> cos(const dual<L>& a) {
    const double s = std::sin(a.v);
    const double c = std::cos(a.v);
    return dual<L>::apply(a, c, -s, -c);
}

template <class L>
inline dual<L> pow(const dual<L>& a, double n) {
    const double f2 = std::pow(a.v, n - 2.0);
    const double f1 = f2 * a.v;
    return dual<L>::apply(a, f1 * a.v, n * f1, n * (n - 1.0) * f2);
}

// The number type of the probe.
using hnumber = dual<ProbeLayout>;

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_DUAL_H
