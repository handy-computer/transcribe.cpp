// arch/nemotron3_diar/torch_logf.h - bit-exact float32 natural log as computed
// by PyTorch on aarch64 CPU (torch.log -> Vectorized<float>::log ->
// Sleef_logf4_u10, AdvSIMD with FMA). Scalar port of Sleef's xlogf_u1
// (refs/sleef src/libm/sleefsimdsp.c + src/common/df.h, ENABLE_FMA_SP).
//
// Why: the AOSC cache compression selects the top spkcache_len frames by a
// log-probability score, and the scores at the selection boundary are
// routinely ~1e-7 apart (and often exactly tied). A 1-ulp difference in log
// (Apple libm disagrees with Sleef on ~0.4% of inputs) flips which frame is
// kept and every later chunk diverges from the NeMo reference. Verified
// bit-identical to torch.log on 252k inputs from the reference dumps.
//
// Sleef is Copyright Naoki Shibata and contributors, distributed under the
// Boost Software License 1.0 (https://www.boost.org/LICENSE_1_0.txt).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

// Sleef keeps every multiply/add separate unless it asks for an FMA.
#pragma clang fp contract(off)

namespace transcribe::nemotron3_diar::torch_logf {

struct f2 {
    float x, y;
};

inline float fma_(float x, float y, float z) {
    return std::fma(x, y, z);
}  // z + x*y

inline float fmapn(float x, float y, float z) {
    return std::fma(x, y, -z);
}  // x*y - z

inline float fmanp(float x, float y, float z) {
    return std::fma(-x, y, z);
}  // z - x*y

inline float add3(float a, float b, float c) {
    return (a + b) + c;
}

inline float add4(float a, float b, float c, float d) {
    return add3(a + b, c, d);
}

inline f2 dfmul_f2_f(f2 x, float y) {
    float s = x.x * y;
    return { s, fma_(x.y, y, fmapn(x.x, y, s)) };
}

inline f2 dfadd2_f_f(float x, float y) {
    float s = x + y;
    float v = s - x;
    return { s, (x - (s - v)) + (y - v) };
}

inline f2 dfdiv(f2 n, f2 d) {
    float t = 1.0f / d.x;
    float s = n.x * t;
    float u = fmapn(t, n.x, s);
    float v = fmanp(d.y, t, fmanp(d.x, t, 1.0f));
    return { s, fma_(s, v, fma_(n.y, t, u)) };
}

inline f2 dfscale(f2 d, float s) {
    return { d.x * s, d.y * s };
}

inline f2 dfadd_f2_f2(f2 x, f2 y) {
    float s = x.x + y.x;
    return { s, add4(x.x - s, y.x, x.y, y.y) };
}

inline f2 dfadd_f2_f(f2 x, float y) {
    float s = x.x + y;
    return { s, add3(x.x - s, y, x.y) };
}

inline int32_t bits(float f) {
    int32_t i;
    std::memcpy(&i, &f, 4);
    return i;
}

inline float from_bits(int32_t i) {
    float f;
    std::memcpy(&f, &i, 4);
    return f;
}

inline float logf_u10(float d) {
    const bool o  = d < std::numeric_limits<float>::min();
    float      dd = o ? d * (float(INT64_C(1) << 32) * float(INT64_C(1) << 32)) : d;
    int32_t    e  = ((bits(dd * (1.0f / 0.75f)) >> 23) & 0xff) - 0x7f;
    float      m  = from_bits(bits(dd) + ((-e) << 23));
    e             = o ? e - 64 : e;
    f2    s       = dfmul_f2_f({ 0.69314718246459960938f, -1.904654323148236017e-09f }, static_cast<float>(e));
    f2    x       = dfdiv(dfadd2_f_f(-1.0f, m), dfadd2_f_f(1.0f, m));
    float x2      = x.x * x.x;
    float t       = 0.3027294874e+0f;
    t             = fma_(t, x2, 0.3996108174e+0f);
    t             = fma_(t, x2, 0.6666694880e+0f);
    s             = dfadd_f2_f2(s, dfscale(x, 2.0f));
    s             = dfadd_f2_f(s, (x2 * x.x) * t);
    float r       = s.x + s.y;
    if (std::isinf(dd) && dd > 0) {
        r = std::numeric_limits<float>::infinity();
    }
    if (dd < 0 || std::isnan(dd)) {
        r = std::numeric_limits<float>::quiet_NaN();
    }
    if (dd == 0) {
        r = -std::numeric_limits<float>::infinity();
    }
    return r;
}
}  // namespace transcribe::nemotron3_diar::torch_logf
