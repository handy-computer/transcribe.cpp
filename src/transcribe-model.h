// transcribe-model.h - internal base class for the public opaque
// transcribe_model handle.
//
// This header is INTERNAL. The public C ABI in include/transcribe.h only
// forward-declares `struct transcribe_model;`; the real definition lives
// here so per-family code (src/arch/<family>/...) can derive from it.
//
// Layout: hybrid of "minimal base" and "full polymorphism" — the base
// owns the universal data and per-call dispatch goes through the Arch
// trait callbacks. RESUME.md "Decisions still load-bearing" has the
// rationale and the rejected alternatives.
//
//   - The base owns everything that is genuinely invariant across all
//     families: the dispatch token (`arch`), the variant string, the
//     bound runtime backend string, and the public capabilities struct.
//   - The base has a virtual destructor so the central
//     transcribe_model_free() can `delete model;` polymorphically without
//     a per-family free callback.
//   - Per-family subclasses (ParakeetModel, etc.) own everything else:
//     the gguf_context, weights, decoder state. RAII frees them in the
//     subclass destructor.
//
// The language pointer chain (the only mildly tricky part of
// transcribe_capabilities) is centralized in set_languages() so per-family
// load() code never has to get it right.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe {
struct Arch;
class Tokenizer;
} // namespace transcribe

// The public C ABI forward-declares this as `struct transcribe_model;`,
// so the real definition stays in the global namespace and uses the
// `struct` keyword for ABI compatibility with C callers.
struct transcribe_model {
    // Dispatch token. Set by per-family load() to the family's Arch
    // instance. The central dispatcher reads this for arch_string() and
    // for per-call dispatch (init_context, run).
    const transcribe::Arch * arch = nullptr;

    // Identification, both surfaced via the public string accessors.
    // variant is whatever the family decided (loader leaves it empty if
    // stt.variant was absent; the family supplies a default).
    std::string variant;

    // Runtime backend currently bound to this model. Empty string means
    // "no backend bound" — both pre-binding and the model == NULL case
    // collapse to that one stable rule. See the public header for the
    // full semantics. In 2B no real backend is wired, so loaders leave
    // this empty.
    std::string backend;

    // Public capabilities. Per-family load() fills this in directly,
    // calling set_languages() for the languages chain. Immutable after
    // a successful load. Zero-initialized here; the capabilities
    // copy-out accessor overwrites struct_size with the caller's view
    // before returning, so the value of caps.struct_size on the model
    // is irrelevant to external observers.
    transcribe_capabilities caps{};

    // Wall-clock load time in microseconds, captured by per-family
    // load() (start at function entry, stop just before *out_model
    // is set). Surfaced via transcribe_get_timings on any context
    // derived from this model — load time is a model-scoped fact.
    int64_t t_load_us = 0;

    transcribe_model() = default;
    virtual ~transcribe_model();

    transcribe_model(const transcribe_model &)             = delete;
    transcribe_model & operator=(const transcribe_model &) = delete;
    transcribe_model(transcribe_model &&)                  = delete;
    transcribe_model & operator=(transcribe_model &&)      = delete;

    // Optional accessor for an internal Tokenizer instance. The default
    // returns nullptr; per-family models that load a tokenizer override
    // it. Used by internal code (and tests) to inspect the vocabulary
    // without dragging the per-family layout into the central dispatch.
    virtual const transcribe::Tokenizer * tokenizer() const { return nullptr; }

    // Replace the languages list. Stores the strings inside the model
    // (so their c_str() lifetime is bound to the model lifetime), then
    // rebuilds the capabilities pointer chain that the public ABI
    // exposes via transcribe_capabilities::languages. Per-family load()
    // calls this once after deciding the language list.
    void set_languages(std::vector<std::string> langs);

private:
    // Backing storage for the languages chain. Kept private so the only
    // way to mutate it is through set_languages(), which guarantees the
    // capability struct's pointer + count stay in sync.
    std::vector<std::string>  language_storage_;
    std::vector<const char *> language_ptrs_;
};
