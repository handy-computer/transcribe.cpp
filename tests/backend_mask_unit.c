/* backend_mask_unit.c - allowed-backend mask (transcribe_init_backends_ex).
 *
 * The mask is fixed for the life of the process, so each scenario runs in
 * its own process: ctest registers this binary once per mode (argv[1]).
 *
 *   cpu      mask=CPU: only CPU-kind devices register, GPU requests are
 *            unavailable, a different mask is refused, the same one is
 *            accepted, and (Linux) no Vulkan ICD was ever loaded.
 *   default  default params: everything allowed; bad struct_size rejected.
 *   env-cpu  TRANSCRIBE_BACKENDS=cpu narrows a host mask of ALL to CPU.
 *   late     a device query before _ex. Static builds register backends on
 *            that query, so a narrower mask is refused; dynamic builds have
 *            nothing registered yet, so it is accepted.
 *
 * TRANSCRIBE_TEST_BACKEND_DIR names the backend module directory (set by
 * ctest to the build's bin dir; static builds scan it as a no-op).
 *
 * Plain C11 on purpose, like api_smoke.c: uses only the public header.
 */

#include "transcribe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

static int only_cpu_kind_devices(void) {
    const int n = transcribe_device_count();
    for (int i = 0; i < n; ++i) {
        struct transcribe_device_info info;
        transcribe_device_info_init(&info);
        if (transcribe_device_get_info(transcribe_device_get(i), &info) != TRANSCRIBE_OK) {
            return 0;
        }
        if (strcmp(info.kind, "cpu") != 0 && strcmp(info.kind, "accel") != 0) {
            fprintf(stderr, "unexpected device %s (kind %s)\n", info.name, info.kind);
            return 0;
        }
    }
    return n > 0;
}

/* Linux Vulkan ICDs are only loaded by vkCreateInstance / instance
 * enumeration, so their absence from the address space proves the Vulkan
 * backend never initialized. Known ICD libraries: Mesa's libvulkan_<driver>
 * (radeon, lvp, intel, ...), NVIDIA's libGLX_nvidia, AMDVLK's amdvlk64/32.
 * Other vendors' ICDs are not recognized; run_default() detects that case
 * and skips the self-check rather than failing. */
static int vulkan_icd_loaded(void) {
#if defined(__linux__)
    static const char * const k_icd_libs[] = { "libvulkan_", "libGLX_nvidia", "amdvlk" };

    FILE * f = fopen("/proc/self/maps", "r");
    if (f == NULL) {
        return 0;
    }
    char line[4096];
    int  found = 0;
    while (!found && fgets(line, sizeof(line), f) != NULL) {
        for (size_t i = 0; i < sizeof(k_icd_libs) / sizeof(k_icd_libs[0]); ++i) {
            if (strstr(line, k_icd_libs[i]) != NULL) {
                found = 1;
                break;
            }
        }
    }
    fclose(f);
    return found;
#else
    return 0;
#endif
}

static transcribe_status init_with_mask(uint32_t mask) {
    struct transcribe_backend_init_params p;
    transcribe_backend_init_params_init(&p);
    p.artifact_dir     = getenv("TRANSCRIBE_TEST_BACKEND_DIR");
    p.allowed_backends = mask;
    return transcribe_init_backends_ex(&p);
}

static void run_cpu(void) {
    CHECK(init_with_mask(TRANSCRIBE_BACKEND_MASK_CPU) == TRANSCRIBE_OK);
    CHECK(transcribe_allowed_backends() == TRANSCRIBE_BACKEND_MASK_CPU);
    CHECK(only_cpu_kind_devices());
    CHECK(transcribe_backend_available(TRANSCRIBE_BACKEND_CPU));
    CHECK(!transcribe_backend_available(TRANSCRIBE_BACKEND_VULKAN));
    CHECK(!transcribe_backend_available(TRANSCRIBE_BACKEND_METAL));
    CHECK(!transcribe_backend_available(TRANSCRIBE_BACKEND_CUDA));
    CHECK(!vulkan_icd_loaded());

    /* Fixed: a different effective mask is refused, the same one (including
     * 0, which is CPU-implied) is accepted, and the legacy entry points keep
     * the fixed mask. */
    CHECK(init_with_mask(TRANSCRIBE_BACKEND_MASK_ALL) == TRANSCRIBE_ERR_BACKEND);
    CHECK(init_with_mask(TRANSCRIBE_BACKEND_MASK_CPU) == TRANSCRIBE_OK);
    CHECK(init_with_mask(0) == TRANSCRIBE_OK);
    CHECK(transcribe_init_backends(getenv("TRANSCRIBE_TEST_BACKEND_DIR")) == TRANSCRIBE_OK);
    CHECK(transcribe_allowed_backends() == TRANSCRIBE_BACKEND_MASK_CPU);
    CHECK(only_cpu_kind_devices());
    CHECK(!vulkan_icd_loaded());
}

static void run_default(void) {
    struct transcribe_backend_init_params bad;
    transcribe_backend_init_params_init(&bad);
    bad.struct_size = 8;
    CHECK(transcribe_init_backends_ex(&bad) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    struct transcribe_backend_init_params p;
    transcribe_backend_init_params_init(&p);
    p.artifact_dir = getenv("TRANSCRIBE_TEST_BACKEND_DIR");
    CHECK(transcribe_init_backends_ex(&p) == TRANSCRIBE_OK);
    CHECK(transcribe_allowed_backends() == TRANSCRIBE_BACKEND_MASK_ALL);
    CHECK(transcribe_device_count() > 0);
#if defined(__linux__)
    /* Probe self-check: the ICD scan should see a Vulkan backend that did
     * initialize, or its absence in the other modes proves nothing. An
     * unrecognized ICD library is a limitation of the probe, not a mask
     * failure, so report it and move on. */
    if (transcribe_backend_available(TRANSCRIBE_BACKEND_VULKAN) && !vulkan_icd_loaded()) {
        fprintf(stderr,
                "note: Vulkan is up but no known ICD library is mapped; the "
                "ICD-absence checks in the other modes are not meaningful on this host\n");
    }
#endif
}

static void run_env_cpu(void) {
    /* ctest sets TRANSCRIBE_BACKENDS=cpu for this mode. */
    CHECK(init_with_mask(TRANSCRIBE_BACKEND_MASK_ALL) == TRANSCRIBE_OK);
    CHECK(transcribe_allowed_backends() == TRANSCRIBE_BACKEND_MASK_CPU);
    CHECK(only_cpu_kind_devices());
    CHECK(!vulkan_icd_loaded());
}

static void run_late(void) {
    const int               pre = transcribe_device_count();
    const transcribe_status st  = init_with_mask(TRANSCRIBE_BACKEND_MASK_CPU);
    if (pre > 0) {
        /* Static build: the query already registered everything under ALL,
         * so narrowing now is refused. */
        CHECK(st == TRANSCRIBE_ERR_BACKEND);
        CHECK(transcribe_allowed_backends() == TRANSCRIBE_BACKEND_MASK_ALL);
    } else {
        CHECK(st == TRANSCRIBE_OK);
        CHECK(transcribe_allowed_backends() == TRANSCRIBE_BACKEND_MASK_CPU);
        CHECK(only_cpu_kind_devices());
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s cpu|default|env-cpu|late\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char * mode = argv[1];
    if (strcmp(mode, "cpu") == 0) {
        run_cpu();
    } else if (strcmp(mode, "default") == 0) {
        run_default();
    } else if (strcmp(mode, "env-cpu") == 0) {
        run_env_cpu();
    } else if (strcmp(mode, "late") == 0) {
        run_late();
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return EXIT_FAILURE;
    }
    if (g_failures > 0) {
        fprintf(stderr, "backend_mask_unit %s: %d failure(s)\n", mode, g_failures);
        return EXIT_FAILURE;
    }
    printf("backend_mask_unit %s: OK\n", mode);
    return EXIT_SUCCESS;
}
