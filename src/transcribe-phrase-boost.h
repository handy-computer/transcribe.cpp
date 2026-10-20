// transcribe-phrase-boost.h - token-level phrase boosting for greedy decoders.
//
// The caller's vocabulary terms, tokenized, become an Aho-Corasick automaton
// over token ids. A greedy decoder keeps one automaton state and, before each
// argmax, adds to every candidate token the score that emitting it would
// earn:
//
//   - a token that extends the current partial match earns `score`;
//   - a token that breaks it gives back what the unfinished match earned and
//     may start a new match through the failure links;
//   - a completed term keeps its bonus, so later tokens never take it back.
//
// The blank token never moves the automaton and is never adjusted. With no
// terms the decoder skips every call, so unbiased decoding stays bit-exact.
// This is shallow fusion with a boosting tree (as in icefall's ContextGraph
// and NeMo's phrase boosting), reduced to what one greedy hypothesis needs.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace transcribe {

class PhraseBoost {
  public:
    // Builds the automaton. Empty phrases are ignored; duplicates merge.
    // `score` is in logit units (natural log), applied per matched token after
    // a term's first one: the first piece is often a short, common word start,
    // and a bonus there rewrites ordinary words. A term is completed once the
    // model has begun it, and nothing is started for it.
    void build(const std::vector<std::vector<int32_t>> & phrases, float score);

    bool empty() const { return nodes_.size() <= 1; }

    static constexpr int root = 0;

    // The state after emitting `token` from `state`.
    int next(int state, int32_t token) const;

    // Adds to logits[0, n) the score each non-blank token would earn from
    // `state`. Tokens outside [0, n) in the automaton are ignored.
    void apply(int state, float * logits, int n, int32_t blank) const;

    // The score emitting `token` from `state` earns (what apply adds to it).
    float delta(int state, int32_t token) const;

    struct Edge {
        int32_t token;
        int     target;
        float   delta;
    };

  private:
    struct Node {
        std::vector<std::pair<int32_t, int>> children;  // sorted by token
        int                                  fail   = 0;
        float                                earned = 0.0f;  // bonus a partial match to here has collected
        float                                locked = 0.0f;  // part of `earned` that completed terms keep
    };

    int child(int state, int32_t token) const;

    std::vector<Node>              nodes_;
    std::vector<std::vector<Edge>> edges_;  // per state, sorted by token
    float                          score_ = 0.0f;
};

}  // namespace transcribe
