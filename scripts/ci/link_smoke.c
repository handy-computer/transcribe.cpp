/* Minimal EXTERNAL consumer of an installed transcribe tree.
 *
 * Compiled by scripts/ci/link_smoke.py against `cmake --install` output,
 * with a link line constructed ONLY from lib/transcribe-link.json — this
 * program plus that manifest are the contract a non-CMake consumer (the
 * Rust -sys crate's build.rs, above all) relies on. No model is loaded:
 * the assertions are link + runtime-registry health, which is exactly what
 * a wrong archive order / missing system lib / broken rpath breaks.
 */
#include <stdio.h>
#include <transcribe.h>

int main(int argc, char ** argv) {
    const char * version = transcribe_version();
    printf("version=%s commit=%s\n", version, transcribe_version_commit());
    if (version == NULL || version[0] == '\0') {
        fprintf(stderr, "link-smoke: empty version\n");
        return 1;
    }

    /* Compiled-in backends register without transcribe_init_backends().
     * GGML_BACKEND_DL builds compile in NONE — the driver passes the
     * installed module directory (manifest `module_dir`) as argv[1], the
     * same call a real DL-posture consumer must make. */
    if (argc > 1) {
        transcribe_status st = transcribe_init_backends(argv[1]);
        printf("init_backends(%s) -> %d\n", argv[1], (int) st);
        if (st != TRANSCRIBE_OK) {
            fprintf(stderr, "link-smoke: init_backends failed\n");
            return 1;
        }
    }

    int n = transcribe_backend_device_count();
    printf("devices=%d\n", n);
    if (n < 1) {
        fprintf(stderr, "link-smoke: no registered compute devices\n");
        return 1;
    }
    for (int i = 0; i < n; i++) {
        struct transcribe_backend_device dev;
        transcribe_backend_device_init(&dev);
        if (transcribe_get_backend_device(i, &dev) != TRANSCRIBE_OK) {
            fprintf(stderr, "link-smoke: device %d query failed\n", i);
            return 1;
        }
        printf("device[%d]=%s kind=%s\n", i, dev.name, dev.kind);
    }

    if (!transcribe_backend_available(TRANSCRIBE_BACKEND_CPU)) {
        fprintf(stderr, "link-smoke: CPU backend unavailable\n");
        return 1;
    }

    printf("link-smoke ok\n");
    return 0;
}
