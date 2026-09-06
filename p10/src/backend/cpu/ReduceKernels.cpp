#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "TensorIteratorOps.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

Tensor isnan_cpu(const Tensor& self);

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1;
    inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i)
        inner *= shape[i];
}

template <class AccT, class Step, class Done>
Tensor reduce_dims_impl(const Tensor& self, std::vector<int64_t> dims_in,
                        bool keepdim, DType out_dtype, AccT init, Step step,
                        Done done) {
    const int64_t nd = self.dim();
    std::vector<bool> reduced(static_cast<size_t>(nd), false);
    for (auto& d : dims_in) {
        d = wrap_dim(d, nd);
        reduced[static_cast<size_t>(d)] = true;
    }
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i) {
        if (!reduced[static_cast<size_t>(i)]) out_shape.push_back(self.size(i));
        else if (keepdim) out_shape.push_back(1);
    }
    std::vector<int64_t> strides(static_cast<size_t>(nd), 0);
    int64_t stride = 1;
    for (int64_t i = nd - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= self.size(i);
    }

    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(out_shape, out_dtype, self.device());
    const int64_t out_numel = out.numel();
    if (out_numel == 0) return out;

    std::vector<int64_t> red_dims;
    std::vector<int64_t> red_strides;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[static_cast<size_t>(i)]) {
            red_dims.push_back(i);
            red_strides.push_back(strides[static_cast<size_t>(i)]);
        }
    }
    int64_t total_red = 1;
    for (const int64_t dim : red_dims) total_red *= self.size(dim);

    parallel_for(0, out_numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        std::vector<int64_t> coords(red_dims.size(), 0);
        for (int64_t oi = begin; oi < end; ++oi) {
            int64_t base = 0;
            int64_t rest = oi;
            std::vector<int64_t> out_coords;
            for (int64_t i = 0; i < nd; ++i) {
                if (!reduced[static_cast<size_t>(i)])
                    out_coords.push_back(self.size(i));
            }
            std::vector<int64_t> coord(out_coords.size(), 0);
            for (int64_t i = static_cast<int64_t>(out_coords.size()) - 1;
                 i >= 0; --i) {
                coord[static_cast<size_t>(i)] = rest % out_coords[static_cast<size_t>(i)];
                rest /= out_coords[static_cast<size_t>(i)];
            }
            int64_t coord_index = 0;
            for (int64_t i = 0; i < nd; ++i) {
                if (reduced[static_cast<size_t>(i)]) continue;
                base += coord[static_cast<size_t>(coord_index++)] *
                        strides[static_cast<size_t>(i)];
            }

            AccT acc = init;
            std::fill(coords.begin(), coords.end(), 0);
            for (int64_t c = 0; c < total_red; ++c) {
                int64_t offset = base;
                for (size_t r = 0; r < red_dims.size(); ++r)
                    offset += coords[r] * red_strides[r];
                switch (sc.dtype()) {
#define TP_REDUCE_STEP(ctype, name_) \
    case DType::name_: \
        acc = step(acc, static_cast<double>(sc.data_ptr<ctype>()[offset])); \
        break;
                    TENSORPLAY_FORALL_SCALAR_TYPES(TP_REDUCE_STEP)
#undef TP_REDUCE_STEP
                    default: TP_THROW(TypeError, "reduce: unsupported dtype");
                }
                for (int64_t r = static_cast<int64_t>(red_dims.size()) - 1;
                     r >= 0; --r) {
                    if (++coords[static_cast<size_t>(r)] <
                        self.size(red_dims[static_cast<size_t>(r)]))
                        break;
                    coords[static_cast<size_t>(r)] = 0;
                }
            }

            const double value = done(acc);
            switch (out_dtype) {
#define TP_REDUCE_DONE(ctype, name_) \
    case DType::name_: \
        out.data_ptr<ctype>()[oi] = static_cast<ctype>(value); \
        break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_REDUCE_DONE)
#undef TP_REDUCE_DONE
                default: TP_THROW(TypeError, "reduce: unsupported output dtype");
            }
        }
    });
    return out;
}

std::pair<Tensor, Tensor> mean_var_over_dims(const Tensor& self,
                                             std::vector<int64_t> dims_in,
                                             bool unbiased, bool keepdim) {
    const int64_t nd = self.dim();
    std::vector<int64_t> dims = std::move(dims_in);
    if (dims.empty()) {
        for (int64_t i = 0; i < nd; ++i) dims.push_back(i);
    }
    std::vector<bool> reduced(static_cast<size_t>(nd), false);
    for (auto& d : dims) {
        d = wrap_dim(d, nd);
        reduced[static_cast<size_t>(d)] = true;
    }
    bool all_reduced = true;
    for (const bool value : reduced) all_reduced = all_reduced && value;
    if (all_reduced) {
        for (int64_t i = 0; i < nd; ++i) reduced[static_cast<size_t>(i)] = true;
    }

    std::vector<int64_t> out_sizes;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[static_cast<size_t>(i)]) {
            if (keepdim) out_sizes.push_back(1);
        } else {
            out_sizes.push_back(self.size(i));
        }
    }
    const DType dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Float32;
    Tensor sc = self.to(dt).contiguous();
    Tensor mean = Tensor::empty(out_sizes, dt, self.device());
    Tensor var = Tensor::empty(out_sizes, dt, self.device());
    const int64_t out_numel = mean.numel();

    std::vector<int64_t> strides(static_cast<size_t>(nd), 0);
    int64_t stride = 1;
    for (int64_t i = nd - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= self.size(i);
    }
    std::vector<int64_t> red_dims;
    std::vector<int64_t> red_strides;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[static_cast<size_t>(i)]) {
            red_dims.push_back(i);
            red_strides.push_back(strides[static_cast<size_t>(i)]);
        }
    }
    int64_t n_red = 1;
    for (const int64_t dim : red_dims) n_red *= self.size(dim);
    const double ddof = unbiased && n_red > 1 ? 1.0 : 0.0;

    auto compute = [&](auto* sp, auto* mp, auto* vp) {
        using value_t = std::remove_pointer_t<decltype(sp)>;
        parallel_for(0, out_numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            std::vector<int64_t> coords(red_dims.size(), 0);
            std::vector<int64_t> out_coords(out_sizes.size(), 0);
            for (int64_t oi = begin; oi < end; ++oi) {
                int64_t rest = oi;
                for (int64_t i = static_cast<int64_t>(out_sizes.size()) - 1;
                     i >= 0; --i) {
                    out_coords[static_cast<size_t>(i)] =
                        rest % out_sizes[static_cast<size_t>(i)];
                    rest /= out_sizes[static_cast<size_t>(i)];
                }
                int64_t base = 0;
                int64_t out_index = 0;
                for (int64_t i = 0; i < nd; ++i) {
                    if (reduced[static_cast<size_t>(i)]) continue;
                    base += out_coords[static_cast<size_t>(out_index++)] *
                            strides[static_cast<size_t>(i)];
                }
                double sum = 0.0;
                double square_sum = 0.0;
                std::fill(coords.begin(), coords.end(), 0);
                for (int64_t c = 0; c < n_red; ++c) {
                    int64_t offset = base;
                    for (size_t r = 0; r < red_dims.size(); ++r)
                        offset += coords[r] * red_strides[r];
                    const double value = static_cast<double>(sp[offset]);
                    sum += value;
                    square_sum += value * value;
                    for (int64_t r = static_cast<int64_t>(red_dims.size()) - 1;
                         r >= 0; --r) {
                        if (++coords[static_cast<size_t>(r)] <
                            self.size(red_dims[static_cast<size_t>(r)]))
                            break;
                        coords[static_cast<size_t>(r)] = 0;
                    }
                }
                const double m = sum / n_red;
                mp[oi] = static_cast<value_t>(m);
                const double variance =
                    (square_sum - m * m * n_red) / (n_red - ddof);
                vp[oi] = static_cast<value_t>(variance > 0.0 ? variance : 0.0);
            }
        });
    };

    if (dt == DType::Float64)
        compute(sc.data_ptr<double>(), mean.data_ptr<double>(), var.data_ptr<double>());
    else
        compute(sc.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>());
    return {var, mean};
}

struct LseState {
    double m;
    double s;
    bool nan_flag;
};

static void zero_numel_check_dims(const Tensor& self,
                                  const std::vector<int64_t>& dims,
                                  const char* fn_name) {
    if (dims.empty()) {
        TP_THROW(RuntimeError, fn_name,
                 ": Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    const int64_t nd = self.dim();
    for (int64_t d : dims) {
        if (d < 0) d += nd;
        TP_CHECK_INDEX(self.size(d) != 0, fn_name,
                       ": Expected reduction dim ", d,
                       " to have non-zero size.");
    }
}

Tensor amax_cpu(const Tensor& self, const std::vector<int64_t>& dim_in,
                bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim_in, "amax()");
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    return reduce_dims_impl<double>(
        self, resolved, keepdim, self.dtype(),
        -std::numeric_limits<double>::infinity(),
        [](double acc, double value) {
            return (value != value || value > acc) ? value : acc;
        },
        [](double acc) { return acc; });
}

Tensor amin_cpu(const Tensor& self, const std::vector<int64_t>& dim_in,
                bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim_in, "amin()");
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    return reduce_dims_impl<double>(
        self, resolved, keepdim, self.dtype(),
        std::numeric_limits<double>::infinity(),
        [](double acc, double value) {
            return (value != value || value < acc) ? value : acc;
        },
        [](double acc) { return acc; });
}

std::tuple<Tensor, Tensor> aminmax_cpu(
    const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    if (self.numel() == 0) {
        if (dim_in.empty()) {
            TP_THROW(RuntimeError,
                     "aminmax(): cannot compute aminmax over an empty dimension as "
                     "the operation has no identity.");
        }
        zero_numel_check_dims(self, dim_in, "aminmax");
    }
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    return {amin_cpu(self, resolved, keepdim),
            amax_cpu(self, resolved, keepdim)};
}

std::tuple<Tensor, Tensor> aminmax_all_cpu(const Tensor& self) {
    return aminmax_cpu(self, {}, false);
}

std::tuple<Tensor, Tensor> aminmax_dim_cpu(const Tensor& self, int64_t dim,
                                           bool keepdim) {
    return aminmax_cpu(self, {dim}, keepdim);
}

Tensor logsumexp_cpu(const Tensor& self, int64_t dim, bool keepdim) {
    if (!isFloatingType(self.dtype()))
        TP_THROW(RuntimeError, "logsumexp(): Expected floating point type");
    LseState init{-std::numeric_limits<double>::infinity(), 0.0, false};
    return reduce_dims_impl<LseState>(
        self, {dim}, keepdim, self.dtype(), init,
        [](LseState acc, double value) {
            if (value != value) {
                acc.nan_flag = true;
                return acc;
            }
            if (acc.m == -std::numeric_limits<double>::infinity()) {
                acc.m = value;
                acc.s = 1.0;
                return acc;
            }
            if (value > acc.m) {
                acc.s = acc.s * std::exp(acc.m - value) + 1.0;
                acc.m = value;
            } else {
                acc.s += std::exp(value - acc.m);
            }
            return acc;
        },
        [](LseState acc) {
            if (acc.nan_flag) return std::numeric_limits<double>::quiet_NaN();
            if (acc.m == -std::numeric_limits<double>::infinity()) return acc.m;
            return acc.m + std::log(acc.s);
        });
}

Tensor nansum_cpu(const Tensor& self, const std::vector<int64_t>& dim_in,
                  bool keepdim) {
    if (isComplexType(self.dtype())) {
        TP_THROW(RuntimeError, "nansum on CPU does not support complex inputs");
    }
    const DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    std::vector<int64_t> dim = dim_in;
    if (dim.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) dim.push_back(i);
    }
    return reduce_dims_impl<double>(
        self, dim, keepdim, out_dt, 0.0,
        [](double acc, double value) {
            return (value != value) ? acc : acc + value;
        },
        [](double acc) { return acc; });
}

std::tuple<Tensor, Tensor> cummax_cpu(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                DType::Int64, sc.device());
    const int64_t d_size = sc.size(dim);
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
#define TP_CUMMAX_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                ctype* dst = vp + o * d_size * inner + in2; \
                int64_t* out_i = ip + o * d_size * inner + in2; \
                ctype best = src[0]; \
                int64_t best_i = 0; \
                dst[0] = best; out_i[0] = 0; \
                for (int64_t j = 1; j < d_size; ++j) { \
                    if (src[j * inner] > best) { best = src[j * inner]; best_i = j; } \
                    dst[j * inner] = best; out_i[j * inner] = best_i; \
                } \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CUMMAX_CASE)
        default: TP_THROW(TypeError, "cummax: unsupported dtype");
    }
#undef TP_CUMMAX_CASE
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> cummin_cpu(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                DType::Int64, sc.device());
    const int64_t d_size = sc.size(dim);
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
#define TP_CUMMIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                ctype* dst = vp + o * d_size * inner + in2; \
                int64_t* out_i = ip + o * d_size * inner + in2; \
                ctype best = src[0]; \
                int64_t best_i = 0; \
                dst[0] = best; out_i[0] = 0; \
                for (int64_t j = 1; j < d_size; ++j) { \
                    if (src[j * inner] < best) { best = src[j * inner]; best_i = j; } \
                    dst[j * inner] = best; out_i[j * inner] = best_i; \
                } \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CUMMIN_CASE)
        default: TP_THROW(TypeError, "cummin: unsupported dtype");
    }
#undef TP_CUMMIN_CASE
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> std_mean_cpu(const Tensor& self,
                                        std::vector<int64_t> dim,
                                        bool unbiased, bool keepdim) {
    auto vr_mean = mean_var_over_dims(self, std::move(dim), unbiased, keepdim);
    return {vr_mean.first.sqrt(), vr_mean.second};
}

std::tuple<Tensor, Tensor> var_mean_cpu(const Tensor& self,
                                        std::vector<int64_t> dim,
                                        bool unbiased, bool keepdim) {
    return mean_var_over_dims(self, std::move(dim), unbiased, keepdim);
}

Tensor nanmedian_cpu(const Tensor& self) {
    Tensor flat = self.to(isFloatingType(self.dtype()) ? DType::Float64
                                                        : DType::Int64)
                      .reshape({self.numel()});
    std::vector<double> values;
    values.reserve(flat.numel());
    const double* data = flat.data_ptr<double>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        if (!(data[i] != data[i])) values.push_back(data[i]);
    }
    const DType out_dtype = isFloatingType(self.dtype()) ? self.dtype()
                                                          : DType::Int64;
    if (values.empty()) {
        Tensor out = Tensor::zeros({}, out_dtype, self.device());
        if (isFloatingType(out_dtype))
            out.fill_(Scalar(std::numeric_limits<double>::quiet_NaN()));
        else
            out.fill_(Scalar(std::numeric_limits<int64_t>::min()));
        return out;
    }
    std::sort(values.begin(), values.end());
    return Tensor::zeros({}, out_dtype, self.device())
        .fill_(Scalar(values[(values.size() - 1) / 2]));
}

std::tuple<Tensor, Tensor> nanmedian_dim_cpu(const Tensor& self, int64_t dim,
                                             bool keepdim) {
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0,
             "nanmedian(): expects a tensor with at least one dimension");
    dim = wrap_dim(dim, nd);
    TP_CHECK(isFloatingType(self.dtype()),
             "nanmedian(): only floating point dtypes are supported");
    Tensor sc = self.contiguous();
    const int64_t d_size = sc.size(dim);
    TP_CHECK(d_size > 0, "nanmedian(): Expected reduction dim ", dim,
             " to have non-zero size");
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i)
        out_shape.push_back(i == dim ? 1 : sc.size(i));
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, sc.device());

#define TP_NANMEDIAN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = values.data_ptr<ctype>(); \
        int64_t* ip = indices.data_ptr<int64_t>(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            std::vector<std::pair<ctype, int64_t>> buf( \
                static_cast<size_t>(std::max<int64_t>(d_size, 1))); \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                int64_t valid = 0; \
                for (int64_t j = 0; j < d_size; ++j) { \
                    const ctype value = src[j * inner]; \
                    if (value != value) continue; \
                    buf[static_cast<size_t>(valid++)] = {value, j}; \
                } \
                const int64_t oi = keepdim ? si : (o * inner + in2); \
                if (valid == 0) { \
                    vp[oi] = std::numeric_limits<ctype>::quiet_NaN(); \
                    ip[oi] = 0; \
                    continue; \
                } \
                std::sort(buf.begin(), buf.begin() + valid, \
                          [](auto& a, auto& b) { return a.first < b.first; }); \
                vp[oi] = buf[static_cast<size_t>((valid - 1) / 2)].first; \
                ip[oi] = buf[static_cast<size_t>((valid - 1) / 2)].second; \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TP_NANMEDIAN_CASE(float, Float32)
        TP_NANMEDIAN_CASE(double, Float64)
        default: TP_THROW(TypeError, "nanmedian: unsupported dtype ",
                          toString(sc.dtype()));
    }
#undef TP_NANMEDIAN_CASE
    return {values, indices};
}

std::tuple<Tensor, Tensor> nanmedian_dim_values_cpu(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& values,
    Tensor& indices) {
    auto result = nanmedian_dim_cpu(self, dim, keepdim);
    values = std::get<0>(result);
    indices = std::get<1>(result);
    return {values, indices};
}

std::tuple<Tensor, Tensor> mode_cpu(const Tensor& self, int64_t dim,
                                    bool keepdim) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar mode input: ", dim);
        }
        Tensor values = Tensor::empty({}, self.dtype(), self.device());
        Tensor indices = Tensor::zeros({}, DType::Int64, self.device());
        values.copy_(self);
        return {values, indices};
    }
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    const int64_t d_size = sc.size(dim);
    if (d_size == 0) {
        TP_THROW(RuntimeError,
                 "mode(): Expected reduction dim ", dim,
                 " to have non-zero size");
    }
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i)
        out_shape.push_back(i == dim ? 1 : sc.size(i));
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, sc.device());
#define TP_MODE_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = values.data_ptr<ctype>(); \
        int64_t* ip = indices.data_ptr<int64_t>(); \
        for (int64_t si = 0; si < outer * inner; ++si) { \
            const int64_t o = si / inner, in2 = si % inner; \
            const ctype* src = sp + o * d_size * inner + in2; \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            for (int64_t j = 0; j < d_size; ++j) buf[j] = {src[j * inner], j}; \
            std::sort(buf.begin(), buf.end(), [](const auto& a, const auto& b) { \
                if (!(a.first < b.first) && !(b.first < a.first)) return a.second < b.second; \
                return a.first < b.first; \
            }); \
            ctype best_value = buf[0].first; \
            int64_t best_count = 0, best_index = buf[0].second, run = 0; \
            for (int64_t j = 0; j < d_size; ++j) { \
                const bool same = j > 0 && !(buf[j].first < buf[j - 1].first) && \
                                  !(buf[j - 1].first < buf[j].first); \
                run = same ? run + 1 : 1; \
                if (run > best_count) { \
                    best_count = run; best_value = buf[j].first; best_index = buf[j].second; \
                } \
            } \
            vp[si] = best_value; ip[si] = best_index; \
        } \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODE_CASE)
        default: TP_THROW(TypeError, "mode: unsupported dtype");
    }
#undef TP_MODE_CASE
    return {values, indices};
}

std::tuple<Tensor, Tensor> kthvalue_cpu(const Tensor& self, int64_t k,
                                        int64_t dim, bool keepdim) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar kthvalue input: ", dim);
        }
        if (k != 1) {
            TP_THROW(RuntimeError,
                     "kthvalue(): selected number k out of range for dim 0");
        }
        Tensor values = Tensor::empty({}, self.dtype(), self.device());
        Tensor indices = Tensor::zeros({}, DType::Int64, self.device());
        values.copy_(self);
        return {values, indices};
    }
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    const int64_t d_size = sc.size(dim);
    if (k < 1 || k > d_size)
        TP_THROW(RuntimeError, "kthvalue(): selected number k out of range for dim ", dim);
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i)
        out_shape.push_back(i == dim ? 1 : sc.size(i));
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, sc.device());
#define TP_KTH_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = values.data_ptr<ctype>(); \
        int64_t* ip = indices.data_ptr<int64_t>(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                for (int64_t j = 0; j < d_size; ++j) buf[j] = {src[j * inner], j}; \
                std::stable_sort(buf.begin(), buf.end(), [](auto& a, auto& b) { return a.first < b.first; }); \
                const int64_t oi = keepdim ? si : (o * inner + in2); \
                vp[oi] = buf[static_cast<size_t>(k - 1)].first; \
                ip[oi] = buf[static_cast<size_t>(k - 1)].second; \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_KTH_CASE)
        default: TP_THROW(TypeError, "kthvalue: unsupported dtype");
    }
#undef TP_KTH_CASE
    return {values, indices};
}

Tensor count_nonzero_cpu(const Tensor& self, const std::vector<int64_t>& dim) {
    if (dim.empty()) {
        int64_t count = 0;
        Tensor sc = self.contiguous();
        for (int64_t i = 0; i < sc.numel(); ++i) {
            bool nonzero = false;
            switch (sc.dtype()) {
#define TP_COUNT_CASE(ctype, name_) \
    case DType::name_: nonzero = static_cast<bool>(sc.data_ptr<ctype>()[i]); break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_COUNT_CASE)
#undef TP_COUNT_CASE
                default: break;
            }
            if (nonzero) ++count;
        }
        return Tensor::zeros({}, DType::Int64, self.device()).fill_(Scalar(count));
    }
    return reduce_dims_impl<double>(
        self, dim, false, DType::Int64, 0.0,
        [](double acc, double value) { return value != 0.0 ? acc + 1.0 : acc; },
        [](double acc) { return acc; });
}

Tensor dist_cpu(const Tensor& self, const Tensor& other, Scalar p) {
    const double pd = p.toDouble();
    Tensor a = self.to(DType::Float64).contiguous();
    Tensor b = other.to(DType::Float64)
                   .expand(broadcast_shapes(
                       static_cast<std::vector<int64_t>>(self.shape()),
                       static_cast<std::vector<int64_t>>(other.shape())))
                   .to(DType::Float64)
                   .contiguous();
    const int64_t n = a.numel();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    double result = 0.0;
    if (pd == std::numeric_limits<double>::infinity()) {
        for (int64_t i = 0; i < n; ++i)
            result = std::max(result, std::fabs(ap[i] - bp[i]));
    } else if (pd == -std::numeric_limits<double>::infinity()) {
        result = std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < n; ++i)
            result = std::min(result, std::fabs(ap[i] - bp[i]));
    } else if (pd == 0.0) {
        for (int64_t i = 0; i < n; ++i)
            if (ap[i] != bp[i]) result += 1.0;
    } else {
        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i)
            sum += std::pow(std::fabs(ap[i] - bp[i]), pd);
        result = std::pow(sum, 1.0 / pd);
    }
    DType out_dtype = promoteTypes(self.dtype(), other.dtype());
    if (!isFloatingType(out_dtype)) out_dtype = DType::Float32;
    return Tensor::zeros({}, out_dtype, self.device())
        .fill_(Scalar(result));
}

Tensor renorm_cpu(const Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    const double pd = p.toDouble();
    const double max_norm = maxnorm.toDouble();
    Tensor sc = self.to(DType::Float64).contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                DType::Float64, sc.device());
    const int64_t d_size = sc.size(dim);
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    const double* sp = sc.data_ptr<double>();
    double* dp = out.data_ptr<double>();
    const int64_t slice_numel = outer * inner;
    parallel_for(0, d_size, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t j = begin; j < end; ++j) {
            double norm = 0.0;
            if (pd == std::numeric_limits<double>::infinity()) {
                for (int64_t si = 0; si < slice_numel; ++si) {
                    const int64_t o = si / inner, in2 = si % inner;
                    norm = std::max(norm, std::fabs(
                        sp[(o * d_size + j) * inner + in2]));
                }
            } else {
                double sum = 0.0;
                for (int64_t si = 0; si < slice_numel; ++si) {
                    const int64_t o = si / inner, in2 = si % inner;
                    sum += std::pow(std::fabs(
                        sp[(o * d_size + j) * inner + in2]), pd);
                }
                norm = std::pow(sum, 1.0 / pd);
            }
            const double factor = norm > max_norm ? max_norm / norm : 1.0;
            for (int64_t si = 0; si < slice_numel; ++si) {
                const int64_t o = si / inner, in2 = si % inner;
                dp[(o * d_size + j) * inner + in2] =
                    sp[(o * d_size + j) * inner + in2] * factor;
            }
        }
    });
    return out.to(self.dtype());
}

Tensor nanmean_cpu(const Tensor& self, std::optional<int64_t> dim_opt,
                  bool keepdim, std::optional<DType> dtype) {
    DType acc_dt = dtype.value_or(DType::Undefined);
    if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
        TP_THROW(TypeError,
                 "nanmean(): expected input to have floating point or complex dtype but got ",
                 toString(self.dtype()));
    }
    if (acc_dt != DType::Undefined && !isFloatingType(acc_dt) &&
        !isComplexType(acc_dt)) {
        TP_THROW(TypeError,
                 "nanmean(): could not infer output dtype. Optional dtype must be either a floating point or complex dtype. Got: ",
                 toString(acc_dt));
    }
    Tensor x = self;
    const bool complex_output_from_real =
        acc_dt != DType::Undefined && isComplexType(acc_dt) &&
        !isComplexType(x.dtype());
    if (acc_dt != DType::Undefined && x.dtype() != acc_dt &&
        !complex_output_from_real) {
        x = x.to(acc_dt);
    } else if (isReducedFloatingType(x.dtype()) && acc_dt == DType::Undefined) {
        x = x.to(DType::Float32);
    }
    std::vector<int64_t> dims;
    if (dim_opt.has_value()) {
        dims.push_back(*dim_opt);
    } else {
        for (int64_t i = 0; i < x.dim(); ++i) dims.push_back(i);
    }
    Tensor total = nansum_cpu(x, dims, keepdim);
    Tensor valid = isnan_cpu(x).logical_not();
    Tensor count = reduce_dims_impl<double>(
        valid, dims, keepdim, DType::Float32, 0.0,
        [](double acc, double value) { return acc + value; },
        [](double acc) { return acc; });
    Tensor zero = count.eq(Scalar(0.0));
    Tensor quot = total.div(count);
    return quot.masked_fill(
        zero, Scalar(std::numeric_limits<double>::quiet_NaN()))
        .to(acc_dt != DType::Undefined
                ? acc_dt
                : (isComplexType(self.dtype()) ? self.dtype() : total.dtype()));
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, ReduceKernels) {
    m.impl("amax", amax_cpu);
    m.impl("amin", amin_cpu);
    m.impl("aminmax", aminmax_cpu);
    m.impl("_aminmax", aminmax_all_cpu);
    m.impl("_aminmax.dim", aminmax_dim_cpu);
    m.impl("logsumexp", logsumexp_cpu);
    m.impl("nansum", nansum_cpu);
    m.impl("nanmedian", nanmedian_cpu);
    m.impl("nanmedian.dim", nanmedian_dim_cpu);
    m.impl("nanmedian.dim_values", nanmedian_dim_values_cpu);
    m.impl("cummax", cummax_cpu);
    m.impl("cummin", cummin_cpu);
    m.impl("std_mean", std_mean_cpu);
    m.impl("var_mean", var_mean_cpu);
    m.impl("mode", mode_cpu);
    m.impl("kthvalue", kthvalue_cpu);
    m.impl("count_nonzero", count_nonzero_cpu);
    m.impl("dist", dist_cpu);
    m.impl("renorm", renorm_cpu);
    m.impl("nanmean", nanmean_cpu);
}
}  // namespace cpu
}  // namespace tensorplay
