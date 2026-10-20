// transcribe-phrase-boost.cpp - see transcribe-phrase-boost.h.

#include "transcribe-phrase-boost.h"

#include <algorithm>
#include <deque>

namespace transcribe {

namespace {

bool by_token(const PhraseBoost::Edge & e, int32_t token) {
    return e.token < token;
}

}  // namespace

int PhraseBoost::child(int state, int32_t token) const {
    const auto & c  = nodes_[static_cast<size_t>(state)].children;
    const auto   it = std::lower_bound(c.begin(), c.end(), token,
                                       [](const std::pair<int32_t, int> & p, int32_t t) { return p.first < t; });
    return (it != c.end() && it->first == token) ? it->second : -1;
}

void PhraseBoost::build(const std::vector<std::vector<int32_t>> & phrases, float score) {
    nodes_.assign(1, Node{});
    edges_.clear();
    score_ = score;
    std::vector<bool> end(1, false);

    // The trie: one node per distinct token prefix.
    for (const std::vector<int32_t> & phrase : phrases) {
        if (phrase.empty()) {
            continue;
        }
        int state = root;
        for (const int32_t token : phrase) {
            int next = child(state, token);
            if (next < 0) {
                next = static_cast<int>(nodes_.size());
                Node node;
                node.earned = nodes_[static_cast<size_t>(state)].earned + (state == root ? 0.0f : score);
                nodes_.push_back(node);
                end.push_back(false);
                auto & c = nodes_[static_cast<size_t>(state)].children;
                c.insert(std::lower_bound(c.begin(), c.end(), std::make_pair(token, -1)), std::make_pair(token, next));
            }
            state = next;
        }
        end[static_cast<size_t>(state)] = true;
    }

    // Failure links and kept bonuses, breadth first so a node's parent and
    // its failure target are always finished before it.
    std::deque<int> queue;
    for (const auto & [token, c] : nodes_[root].children) {
        nodes_[static_cast<size_t>(c)].fail   = root;
        nodes_[static_cast<size_t>(c)].locked = end[static_cast<size_t>(c)] ? nodes_[static_cast<size_t>(c)].earned : 0.0f;
        queue.push_back(c);
    }
    while (!queue.empty()) {
        const int u = queue.front();
        queue.pop_front();
        for (const auto & [token, v] : nodes_[static_cast<size_t>(u)].children) {
            int f = nodes_[static_cast<size_t>(u)].fail;
            while (f != root && child(f, token) < 0) {
                f = nodes_[static_cast<size_t>(f)].fail;
            }
            const int target                      = child(f, token);
            nodes_[static_cast<size_t>(v)].fail   = (target >= 0 && target != v) ? target : root;
            nodes_[static_cast<size_t>(v)].locked = end[static_cast<size_t>(v)] ? nodes_[static_cast<size_t>(v)].earned :
                                                                                  nodes_[static_cast<size_t>(u)].locked;
            queue.push_back(v);
        }
    }

    // Every transition out of every state, resolved through the failure
    // chain once here instead of on each decode step. The deepest match wins.
    edges_.assign(nodes_.size(), {});
    for (size_t s = 0; s < nodes_.size(); ++s) {
        const Node &        from = nodes_[s];
        std::vector<Edge> & out  = edges_[s];
        for (int f = static_cast<int>(s);; f = nodes_[static_cast<size_t>(f)].fail) {
            for (const auto & [token, target] : nodes_[static_cast<size_t>(f)].children) {
                const auto at = std::lower_bound(out.begin(), out.end(), token, by_token);
                if (at != out.end() && at->token == token) {
                    continue;
                }
                // Extending the match keeps everything earned so far. Falling
                // back gives back the unfinished part and keeps completed
                // terms; the suffix that still matches is earned again.
                const bool  extends = f == static_cast<int>(s);
                const float delta =
                    nodes_[static_cast<size_t>(target)].earned - from.earned + (extends ? 0.0f : from.locked);
                out.insert(at, Edge{ token, target, delta });
            }
            if (f == root) {
                break;
            }
        }
    }
}

int PhraseBoost::next(int state, int32_t token) const {
    const auto & out = edges_[static_cast<size_t>(state)];
    const auto   at  = std::lower_bound(out.begin(), out.end(), token, by_token);
    return (at != out.end() && at->token == token) ? at->target : root;
}

float PhraseBoost::delta(int state, int32_t token) const {
    const auto & out = edges_[static_cast<size_t>(state)];
    const auto   at  = std::lower_bound(out.begin(), out.end(), token, by_token);
    if (at != out.end() && at->token == token) {
        return at->delta;
    }
    const Node & from = nodes_[static_cast<size_t>(state)];
    return from.locked - from.earned;
}

void PhraseBoost::apply(int state, float * logits, int n, int32_t blank) const {
    const Node & from = nodes_[static_cast<size_t>(state)];
    const float  base = from.locked - from.earned;
    if (base != 0.0f) {
        for (int t = 0; t < n; ++t) {
            if (t != blank) {
                logits[t] += base;
            }
        }
    }
    for (const Edge & e : edges_[static_cast<size_t>(state)]) {
        if (e.token >= 0 && e.token < n && e.token != blank) {
            logits[e.token] += e.delta - base;
        }
    }
}

}  // namespace transcribe
