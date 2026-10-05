#include "anacal.h"
#include <sstream>

namespace anacal {
namespace math {

    // A math::dual layout as a Python class: constructed from its flat
    // slots (an array or list in ``L::names`` order) or from a value, the
    // slots readable by name, ``to_array``, and the operators.
    template <class L>
    void bind_dual(py::module_& math, const char* name) {
        using D = dual<L>;
        constexpr int nslot = D::NSLOT;
        const std::string doc =
            std::string("A multi-dual number of the ") + name + " layout.\n"
            "\n"
            "A value, one first-order slot per direction and one\n"
            "second-order slot per pair of directions the layout lists;\n"
            "the arithmetic is that of hyper-dual numbers truncated to\n"
            "those slots, so each slot is an exact derivative of the\n"
            "result.  The slots are readable by name (``names``) and as an\n"
            "array (``to_array``); the operators with another number or a\n"
            "float are defined, as are ``exp, log, sqrt, tanh, pow``.";
        auto cls = py::class_<D>(math, name, doc.c_str())
            .def(py::init<>(), "A number with every slot zero.")
            .def(
                py::init<double>(),
                "A number from its value.\n"
                "\n"
                "Parameters\n"
                "----------\n"
                "v : float\n"
                "    The value; every slot is zero.",
                py::arg("v")
            )
            .def(
                py::init([](const std::array<double, nslot>& a) {
                    return D(a);
                }),
                "A number from its slots.\n"
                "\n"
                "Parameters\n"
                "----------\n"
                "data : array_like, (NSLOT,)\n"
                "    The slots, value first, in ``names`` order.",
                py::arg("data")
            )
            .def_property_readonly_static(
                "names",
                [](py::object) {
                    std::vector<std::string> out(
                        L::names.begin(), L::names.end()
                    );
                    return out;
                },
                "The slot names, value first: list of str."
            )
            .def(
                "to_array",
                [](const D& q) {
                    auto result = py::array_t<double>(nslot);
                    auto r = result.mutable_unchecked<1>();
                    const std::array<double, nslot> s = q.slots();
                    for (int k = 0; k < nslot; ++k) r(k) = s[k];
                    return result;
                },
                "The slots as an array.\n"
                "\n"
                "Returns\n"
                "-------\n"
                "ndarray, (NSLOT,)\n"
                "    The value first, then the slots in ``names`` order."
            )
            .def(
                "shift_reference",
                &D::shift_reference,
                "Move the reference point of the coordinates, in place.\n"
                "\n"
                "The shear slots gain the position-coupling terms of a\n"
                "move by (dx1, dx2), through the translation slots; the\n"
                "pairs holding a shear direction move the same way.\n"
                "\n"
                "Parameters\n"
                "----------\n"
                "dx1, dx2 : float\n"
                "    The move of the reference point, in sky units.",
                py::arg("dx1"), py::arg("dx2")
            )
            .def("__repr__", [name](const D& q) {
                std::ostringstream os;
                os << "<" << name << " " << q << ">";
                return os.str();
            })
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self / py::self)
            .def(py::self + double())
            .def(double() + py::self)
            .def(py::self - double())
            .def(double() - py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(double() / py::self);
        for (int k = 0; k < nslot; ++k) {
            cls.def_property_readonly(
                L::names[k],
                [k](const D& q) { return q.slot(k); },
                (std::string("The slot ") + L::names[k] + ": float.").c_str()
            );
        }
        const std::string fdoc = std::string(
            " of a number, by the chain rule.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "q : ") + name + "\n"
            "    The argument.\n"
            "\n"
            "Returns\n"
            "-------\n" + name + "\n"
            "    The result, every slot an exact derivative.";
        math.def(
            "exp", [](const D& q) { return exp(q); },
            ("The exponential" + fdoc).c_str(), py::arg("q")
        );
        math.def(
            "log", [](const D& q) { return log(q); },
            ("The natural logarithm" + fdoc).c_str(), py::arg("q")
        );
        math.def(
            "sqrt", [](const D& q) { return sqrt(q); },
            ("The square root" + fdoc).c_str(), py::arg("q")
        );
        math.def(
            "tanh", [](const D& q) { return tanh(q); },
            ("The hyperbolic tangent" + fdoc).c_str(), py::arg("q")
        );
        math.def(
            "pow", [](const D& q, double n) { return pow(q, n); },
            (std::string("The power of a number, by the chain rule.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "q : ") + name + "\n"
            "    The base.\n"
            "n : float\n"
            "    The exponent.\n"
            "\n"
            "Returns\n"
            "-------\n" + name + "\n"
            "    ``q ** n``, every slot an exact derivative.").c_str(),
            py::arg("q"), py::arg("n")
        );
    }

    void
    pyExportMath(py::module_ &m) {
        py::module_ math = m.def_submodule(
            "math",
            "submodule for Math"
        );
        py::class_<qnumber>(math, "qnumber")
            .def(py::init<>())
            .def(
                py::init<double>(),
                "initialize qnumber with double number as value",
                py::arg("v")
            )
            .def(
                py::init<double, double, double, double, double>(),
                "initialize qnumber with double numbers",
                py::arg("v"), py::arg("g1"), py::arg("g2"),
                py::arg("x1"), py::arg("x2")
            )
            .def(
                py::init<std::array<double, 5>&>(),
                "initialize qnumber with a list of double numbers",
                py::arg("data")
            )
            .def_readwrite("v", &qnumber::v)
            .def_readwrite("g1", &qnumber::g1)
            .def_readwrite("g2", &qnumber::g2)
            .def_readwrite("x1", &qnumber::x1)
            .def_readwrite("x2", &qnumber::x2)
            .def(
                "to_array",
                &qnumber::to_array,
                "Return the numpy array in shape (5)"
            )
            .def("__repr__", [](
                const qnumber &tn) {
                return "<qnumber v=" + std::to_string(tn.v) +
                       ", g1=" + std::to_string(tn.g1) +
                       ", g2=" + std::to_string(tn.g2) +
                       ", x1=" + std::to_string(tn.x1) +
                       ", x2=" + std::to_string(tn.x2) + ">";
                }
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self / py::self)
            .def(py::self + double())
            .def(double() + py::self)
            .def(py::self - double())
            .def(double() - py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(double() / py::self);
        // the qnumber overloads by signature: the dual templates below
        // share the names
        math.def(
            "exp",
            static_cast<qnumber (*)(const qnumber&)>(&exp),
            "Compute the exponential of a qnumber"
        );
        math.def(
            "pow",
            static_cast<qnumber (*)(const qnumber&, double)>(&pow),
            "Compute the exponential of a qnumber"
        );

        bind_dual<ProbeLayout>(math, "hnumber");

        using size_type = qtensor::size_type;
        py::class_<qtensor>(math, "qtensor")
            .def(py::init<>())
            .def(py::init<const std::vector<size_type>&>())
            .def(py::init<const std::vector<size_type>&, const qnumber&>())
            .def_static(
                "from_flat",
                [](std::vector<qnumber> data,
                   const std::vector<size_type>& shape) {
                    return qtensor::from_vector(std::move(data), shape);
                },
                py::arg("data"),
                py::arg("shape"),
                "Create a qtensor from a flat buffer of qnumbers"
            )
            .def_static(
                "from_image",
                [](std::vector<qnumber> data,
                   size_type height,
                   size_type width) {
                    return qtensor::from_image(
                        std::move(data),
                        height,
                        width
                    );
                },
                py::arg("data"),
                py::arg("height"),
                py::arg("width"),
                "Create a qtensor from ImageQ output"
            )
            .def_property_readonly(
                "shape",
                [](const qtensor& self) { return self.shape(); },
                "Tensor shape expressed as a tuple of dimension lengths"
            )
            .def_property_readonly(
                "strides",
                [](const qtensor& self) { return self.strides(); },
                "Tensor strides in units of qnumbers"
            )
            .def_property_readonly(
                "ndim",
                [](const qtensor& self) { return self.ndim(); },
                "Number of tensor dimensions"
            )
            .def("size", &qtensor::size, "Total number of qnumbers")
            .def("empty", &qtensor::empty, "Return True if the tensor is empty")
            .def(
                "is_contiguous",
                &qtensor::is_contiguous,
                "Return True if the tensor uses contiguous storage"
            )
            .def(
                "reshape",
                &qtensor::reshape,
                py::arg("shape"),
                "Return a reshaped view of the tensor"
            )
            .def(
                "slice",
                &qtensor::slice,
                py::arg("dim"),
                py::arg("start"),
                py::arg("stop"),
                py::arg("step") = 1,
                "Return a view defined by slicing a single dimension"
            )
            .def(
                "select",
                &qtensor::select,
                py::arg("dim"),
                py::arg("index"),
                "Return a view with a dimension fixed to a single index"
            )
            .def(
                "get",
                [](const qtensor& self, const std::vector<size_type>& indices) {
                    return self.at(indices);
                },
                py::arg("indices"),
                "Access a single qnumber using multidimensional indices"
            )
            .def(
                "set",
                [](qtensor& self,
                   const std::vector<size_type>& indices,
                   const qnumber& value) {
                    self.at(indices) = value;
                },
                py::arg("indices"),
                py::arg("value"),
                "Assign a qnumber using multidimensional indices"
            )
            .def(
                "to_list",
                &qtensor::to_vector,
                "Return a flat list of qnumbers in row-major order"
            );

        py::class_<qmatrix<6, 6>>(math, "qmatrix6")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<qnumber, 6>, 6>&>(),
                "initialize qmatrix with a 2D list of qnumber",
                py::arg("data")
            )
            .def(
                py::init<std::array<std::array<double, 6>, 6>&>(),
                "initialize qmatrix with a 2D list of double",
                py::arg("data")
            )
            .def(
                py::init<py::array_t<double>&>(),
                "initialize qmatrix with a 2D numpy array",
                py::arg("data")
            )
            .def_readonly("data", &qmatrix<6, 6>::data)
            .def(
                "transpose",
                &qmatrix<6, 6>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &qmatrix<6, 6>::to_array,
                "Return the numpy array in shape (6, 6, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + qnumber())
            .def(qnumber() + py::self)
            .def(py::self - qnumber())
            .def(qnumber() - py::self)
            .def(py::self * qnumber())
            .def(qnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / qnumber())
            .def(py::self * py::self);

        py::class_<qmatrix<8, 8>>(math, "qmatrix8")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<qnumber, 8>, 8>&>(),
                "initialize qmatrix with a 2D list of qnumber",
                py::arg("data")
            )
            .def(
                py::init<std::array<std::array<double, 8>, 8>&>(),
                "initialize qmatrix with a 2D list of double number",
                py::arg("data")
            )
            .def(
                py::init<py::array_t<double>&>(),
                "initialize qmatrix with a 2D numpy array of double number",
                py::arg("data")
            )
            .def_readonly("data", &qmatrix<8, 8>::data)
            .def(
                "transpose",
                &qmatrix<8, 8>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &qmatrix<8, 8>::to_array,
                "Return the numpy array in shape (8, 8, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + qnumber())
            .def(qnumber() + py::self)
            .def(py::self - qnumber())
            .def(qnumber() - py::self)
            .def(py::self * qnumber())
            .def(qnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / qnumber())
            .def(py::self * py::self);
        math.def("eye6", &eye<6>, "Generate a indentity qmatrix");
        math.def("eye8", &eye<8>, "Generate a indentity qmatrix");

        py::class_<lossNumber> loss_cls(math, "lossNumber");
        loss_cls.def(py::init<>());
#define ANACAL_LOSS_BIND(name) \
        loss_cls.def_readwrite(#name, &lossNumber::name);
        ANACAL_LOSS_FIELDS(ANACAL_LOSS_BIND)
#undef ANACAL_LOSS_BIND
        loss_cls.def_readwrite("n_pix", &lossNumber::n_pix);

    }
} // math
} // anacal
