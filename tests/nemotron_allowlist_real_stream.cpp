#include "transcribe.h"
#include "wav.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_WAV_FILE
#    error "TRANSCRIBE_TEST_WAV_FILE must be defined"
#endif

int main() {
    const char * path = std::getenv("TRANSCRIBE_NEMOTRON_GGUF");
    if (path == nullptr || *path == '\0') {
        return 77;
    }

    transcribe_model_load_params load_params;
    transcribe_model_load_params_init(&load_params);
    transcribe_model * model = nullptr;
    if (transcribe_model_load_file(path, &load_params, &model) != TRANSCRIBE_OK) {
        return 1;
    }
    transcribe_session * session = nullptr;
    if (transcribe_session_init(model, nullptr, &session) != TRANSCRIBE_OK) {
        transcribe_model_free(model);
        return 2;
    }

    std::vector<float> pcm;
    std::string error;
    if (!transcribe_cli::load_wav_mono_16k(TRANSCRIBE_TEST_WAV_FILE, pcm, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        transcribe_session_free(session);
        transcribe_model_free(model);
        return 3;
    }

    transcribe_run_params run;
    transcribe_run_params_init(&run);
    const char * allowed[] = { "en-US", "de-DE" };
    run.allowed_languages = allowed;
    run.n_allowed_languages = 2;
    run.keep_special_tags = true;

    transcribe_status status = transcribe_stream_begin(session, &run, nullptr);
    for (size_t offset = 0; status == TRANSCRIBE_OK && offset < pcm.size(); offset += 16000) {
        const int count = static_cast<int>(std::min<size_t>(16000, pcm.size() - offset));
        status = transcribe_stream_feed(session, pcm.data() + offset, count, nullptr);
    }
    if (status == TRANSCRIBE_OK) {
        status = transcribe_stream_finalize(session, nullptr);
    }
    const std::string raw = status == TRANSCRIBE_OK ? transcribe_raw_text(session) : "";
    if (status != TRANSCRIBE_OK ||
        (raw.find("<en-US>") == std::string::npos && raw.find("<de-DE>") == std::string::npos)) {
        std::fprintf(stderr, "stream failed or emitted no allowed language tag: %s, raw=%s\n",
                     transcribe_status_string(status), raw.c_str());
        status = TRANSCRIBE_ERR_BACKEND;
    }

    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    if (transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK) {
        for (int i = 0; i < caps.n_languages; ++i) {
            const std::string code = caps.languages[i];
            if (code != "en-US" && code != "de-DE" && raw.find("<" + code + ">") != std::string::npos) {
                status = TRANSCRIBE_ERR_BACKEND;
            }
        }
    }

    transcribe_session_free(session);
    transcribe_model_free(model);
    return status == TRANSCRIBE_OK ? 0 : 4;
}
