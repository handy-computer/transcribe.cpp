include(CheckCSourceRuns)

set(AVX_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 a;
        a = _mm256_set1_ps(0);
        return 0;
    }
")

set(AVX512_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0);
        __m512i b = a;
        __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
        return 0;
    }
")

set(AVX2_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256i a = {0};
        a = _mm256_abs_epi16(a);
        __m256i x;
        _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
        return 0;
    }
")

# AVX-512 VNNI: MSVC has no flag to enable only this subset and does not define
# __AVX512VNNI__, so it must be probed and the macro set manually (mirrors how
# ggml-cpu/CMakeLists.txt handles GGML_AVX512_VNNI). check_c_source_runs both
# COMPILES the dpbusd intrinsic (needs /arch:AVX512) and RUNS it (confirms the
# build host actually has VNNI — SIGILL-safe, stays off when absent / cross-building).
set(AVX512_VNNI_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i acc = _mm512_setzero_si512();
        __m512i a   = _mm512_setzero_si512();
        __m512i b   = _mm512_setzero_si512();
        acc = _mm512_dpbusd_epi32(acc, a, b);
        return _mm_cvtsi128_si32(_mm512_castsi512_si128(acc));
    }
")

set(FMA_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 acc = _mm256_setzero_ps();
        const __m256 d = _mm256_setzero_ps();
        const __m256 p = _mm256_setzero_ps();
        acc = _mm256_fmadd_ps( d, p, acc );
        return 0;
    }
")

macro(check_sse type flags)
    set(__FLAG_I 1)
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    foreach (__FLAG ${flags})
        if (NOT ${type}_FOUND)
            set(CMAKE_REQUIRED_FLAGS ${__FLAG})
            check_c_source_runs("${${type}_CODE}" HAS_${type}_${__FLAG_I})
            if (HAS_${type}_${__FLAG_I})
                set(${type}_FOUND TRUE CACHE BOOL "${type} support")
                set(${type}_FLAGS "${__FLAG}" CACHE STRING "${type} flags")
            endif()
            math(EXPR __FLAG_I "${__FLAG_I}+1")
        endif()
    endforeach()
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if (NOT ${type}_FOUND)
        set(${type}_FOUND FALSE CACHE BOOL "${type} support")
        set(${type}_FLAGS "" CACHE STRING "${type} flags")
    endif()

    mark_as_advanced(${type}_FOUND ${type}_FLAGS)
endmacro()

# flags are for MSVC only!
check_sse("AVX" " ;/arch:AVX")
if (NOT ${AVX_FOUND})
    set(GGML_AVX OFF)
else()
    set(GGML_AVX ON)
endif()

check_sse("AVX2" " ;/arch:AVX2")
check_sse("FMA" " ;/arch:AVX2")
if ((NOT ${AVX2_FOUND}) OR (NOT ${FMA_FOUND}))
    set(GGML_AVX2 OFF)
else()
    set(GGML_AVX2 ON)
endif()

check_sse("AVX512" " ;/arch:AVX512")
if (NOT ${AVX512_FOUND})
    set(GGML_AVX512 OFF)
else()
    set(GGML_AVX512 ON)
    # Probe the VNNI subset (dpbusd) only when the AVX-512 base is present.
    # Enabling it lets ggml's int8 dot + the transcribe.cpp Q8_0 blocked GEMM
    # use _mm*_dpbusd_epi32 instead of the slower maddubs+madd path.
    check_sse("AVX512_VNNI" " ;/arch:AVX512")
    if (${AVX512_VNNI_FOUND})
        set(GGML_AVX512_VNNI ON)
    endif()
endif()
