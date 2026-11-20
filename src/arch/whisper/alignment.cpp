#include "alignment.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>

namespace transcribe::whisper {

namespace {

struct alignment_preset {
    std::string_view       variant;
    int                    n_layers;
    int                    n_heads;
    const alignment_head * heads;
    size_t                 count;
};

// OpenAI's alignment-head coordinates, as tabulated in ggml-org/whisper.cpp
// (MIT, Copyright (c) 2023-2024 The ggml authors).
// https://github.com/ggml-org/whisper.cpp/blob/v1.8.3/src/whisper.cpp#L360-L371
constexpr alignment_head k_tiny_en[] = {
    { 1, 0 },
    { 2, 0 },
    { 2, 5 },
    { 3, 0 },
    { 3, 1 },
    { 3, 2 },
    { 3, 3 },
    { 3, 4 },
};
constexpr alignment_head k_tiny[] = {
    { 2, 2 },
    { 3, 0 },
    { 3, 2 },
    { 3, 3 },
    { 3, 4 },
    { 3, 5 },
};
constexpr alignment_head k_base_en[] = {
    { 3, 3 },
    { 4, 7 },
    { 5, 1 },
    { 5, 5 },
    { 5, 7 },
};
constexpr alignment_head k_base[] = {
    { 3, 1 },
    { 4, 2 },
    { 4, 3 },
    { 4, 7 },
    { 5, 1 },
    { 5, 2 },
    { 5, 4 },
    { 5, 6 },
};
constexpr alignment_head k_small_en[] = {
    { 6,  6  },
    { 7,  0  },
    { 7,  3  },
    { 7,  8  },
    { 8,  2  },
    { 8,  5  },
    { 8,  7  },
    { 9,  0  },
    { 9,  4  },
    { 9,  8  },
    { 9,  10 },
    { 10, 0  },
    { 10, 1  },
    { 10, 2  },
    { 10, 3  },
    { 10, 6  },
    { 10, 11 },
    { 11, 2  },
    { 11, 4  },
};
constexpr alignment_head k_small[] = {
    { 5,  3 },
    { 5,  9 },
    { 8,  0 },
    { 8,  4 },
    { 8,  7 },
    { 8,  8 },
    { 9,  0 },
    { 9,  7 },
    { 9,  9 },
    { 10, 5 },
};
constexpr alignment_head k_medium_en[] = {
    { 11, 4  },
    { 14, 1  },
    { 14, 12 },
    { 14, 14 },
    { 15, 4  },
    { 16, 0  },
    { 16, 4  },
    { 16, 9  },
    { 17, 12 },
    { 17, 14 },
    { 18, 7  },
    { 18, 10 },
    { 18, 15 },
    { 20, 0  },
    { 20, 3  },
    { 20, 9  },
    { 20, 14 },
    { 21, 12 },
};
constexpr alignment_head k_medium[] = {
    { 13, 15 },
    { 15, 4  },
    { 15, 15 },
    { 16, 1  },
    { 20, 0  },
    { 23, 4  },
};
constexpr alignment_head k_large_v1[] = {
    { 9,  19 },
    { 11, 2  },
    { 11, 4  },
    { 11, 17 },
    { 22, 7  },
    { 22, 11 },
    { 22, 17 },
    { 23, 2  },
    { 23, 15 },
};
constexpr alignment_head k_large_v2[] = {
    { 10, 12 },
    { 13, 17 },
    { 16, 11 },
    { 16, 12 },
    { 16, 13 },
    { 17, 15 },
    { 17, 16 },
    { 18, 4  },
    { 18, 11 },
    { 18, 19 },
    { 19, 11 },
    { 21, 2  },
    { 21, 3  },
    { 22, 3  },
    { 22, 9  },
    { 22, 12 },
    { 23, 5  },
    { 23, 7  },
    { 23, 13 },
    { 25, 5  },
    { 26, 1  },
    { 26, 12 },
    { 27, 15 },
};
constexpr alignment_head k_large_v3[] = {
    { 7,  0  },
    { 10, 17 },
    { 12, 18 },
    { 13, 12 },
    { 16, 1  },
    { 17, 14 },
    { 19, 11 },
    { 21, 4  },
    { 24, 1  },
    { 25, 6  },
};
constexpr alignment_head k_large_v3_turbo[] = {
    { 2, 4  },
    { 2, 11 },
    { 3, 3  },
    { 3, 6  },
    { 3, 11 },
    { 3, 14 },
};

constexpr alignment_preset k_presets[] = {
    { "whisper-tiny.en",        4,  6,  k_tiny_en,        std::size(k_tiny_en)        },
    { "whisper-tiny",           4,  6,  k_tiny,           std::size(k_tiny)           },
    { "whisper-base.en",        6,  8,  k_base_en,        std::size(k_base_en)        },
    { "whisper-base",           6,  8,  k_base,           std::size(k_base)           },
    { "whisper-small.en",       12, 12, k_small_en,       std::size(k_small_en)       },
    { "whisper-small",          12, 12, k_small,          std::size(k_small)          },
    { "whisper-medium.en",      24, 16, k_medium_en,      std::size(k_medium_en)      },
    { "whisper-medium",         24, 16, k_medium,         std::size(k_medium)         },
    { "whisper-large-v1",       32, 20, k_large_v1,       std::size(k_large_v1)       },
    { "whisper-large-v2",       32, 20, k_large_v2,       std::size(k_large_v2)       },
    { "whisper-large-v3",       32, 20, k_large_v3,       std::size(k_large_v3)       },
    { "whisper-large-v3-turbo", 4,  20, k_large_v3_turbo, std::size(k_large_v3_turbo) },
};

// Whisper smooths each normalized token row along the encoder frame axis.
constexpr int k_median_radius = 3;

float median_at(const std::vector<float> & row, int frame) {
    const int64_t last_frame = static_cast<int64_t>(row.size()) - 1;
    if (last_frame < k_median_radius) {
        return row[frame];
    }

    std::array<float, 2 * k_median_radius + 1> window;
    for (int offset = -k_median_radius; offset <= k_median_radius; ++offset) {
        int64_t index = static_cast<int64_t>(frame) + offset;
        if (index < 0) {
            index = -index;
        } else if (index > last_frame) {
            index = 2 * last_frame - index;
        }
        window[offset + k_median_radius] = row[static_cast<size_t>(index)];
    }
    std::nth_element(window.begin(), window.begin() + k_median_radius, window.end());
    return window[k_median_radius];
}

enum class step : uint8_t { FRAME, TOKEN, BOTH };

void dtw_boundaries(const std::vector<float> & scores,
                    int                        n_rows,
                    int                        n_frames,
                    int                        n_heads,
                    std::vector<int> &         token_frames) {
    // Store one direction per cell and keep only the two live cost rows.
    const size_t        width = static_cast<size_t>(n_frames) + 1;
    const double        inf   = std::numeric_limits<double>::infinity();
    std::vector<double> previous(width, inf);
    std::vector<double> current(width, inf);
    std::vector<step>   trace(scores.size());
    previous[0] = 0.0;

    for (int row = 0; row < n_rows; ++row) {
        current[0] = inf;
        for (int frame = 0; frame < n_frames; ++frame) {
            const size_t index = static_cast<size_t>(row) * n_frames + frame;
            const size_t col   = static_cast<size_t>(frame) + 1;

            // Among minimum-cost steps, prefer a frame advance, then a diagonal.
            double cost      = current[col - 1];
            step   direction = step::FRAME;
            if (previous[col - 1] < cost) {
                cost      = previous[col - 1];
                direction = step::BOTH;
            }
            if (previous[col] < cost) {
                cost      = previous[col];
                direction = step::TOKEN;
            }
            current[col] = cost - static_cast<double>(scores[index]) / n_heads;
            trace[index] = direction;
        }
        std::swap(previous, current);
    }

    token_frames.resize(n_rows);
    int row   = n_rows - 1;
    int frame = n_frames - 1;
    while (true) {
        token_frames[row] = frame;
        if (row == 0 && frame == 0) {
            break;
        }
        const step direction = trace[static_cast<size_t>(row) * n_frames + frame];
        if (direction != step::FRAME) {
            --row;
        }
        if (direction != step::TOKEN) {
            --frame;
        }
    }
}

}  // namespace

std::vector<alignment_head> alignment_heads_for_variant(std::string_view variant, int n_layers, int n_heads) {
    // The converter's original large GGUF uses the v1 checkpoint.
    if (variant == "whisper-large") {
        variant = "whisper-large-v1";
    }
    for (const auto & preset : k_presets) {
        if (variant == preset.variant && n_layers == preset.n_layers && n_heads == preset.n_heads) {
            return { preset.heads, preset.heads + preset.count };
        }
    }
    return {};
}

bool align_attention(const std::vector<float> & attention,
                     int                        n_heads,
                     int                        n_tokens,
                     int                        n_frames,
                     int                        first_alignment_row,
                     int                        n_alignment_rows,
                     std::vector<int> &         token_frames) {
    token_frames.clear();
    if (n_heads <= 0 || n_tokens <= 0 || n_frames <= 0 || first_alignment_row < 0 || n_alignment_rows <= 0 ||
        first_alignment_row > n_tokens || n_alignment_rows > n_tokens - first_alignment_row) {
        return false;
    }
    if (static_cast<size_t>(n_tokens) > attention.size() / n_frames) {
        return false;
    }
    const size_t head_size = static_cast<size_t>(n_tokens) * n_frames;
    if (static_cast<size_t>(n_heads) > attention.size() / head_size || head_size * n_heads != attention.size()) {
        return false;
    }

    // Attention processing follows OpenAI Whisper (MIT, Copyright (c) 2022 OpenAI).
    // https://github.com/openai/whisper/blob/v20250625/whisper/timing.py#L192-L212
    // Accumulate smoothed rows one head at a time to avoid duplicating the input.
    std::vector<float>  scores(static_cast<size_t>(n_alignment_rows) * n_frames, 0.0f);
    std::vector<double> mean(n_frames);
    std::vector<double> inv_std(n_frames);
    std::vector<float>  normalized(n_frames);

    for (int head = 0; head < n_heads; ++head) {
        const float * values = attention.data() + head_size * head;
        std::fill(mean.begin(), mean.end(), 0.0);
        for (int token = 0; token < n_tokens; ++token) {
            const float * row = values + static_cast<size_t>(token) * n_frames;
            for (int frame = 0; frame < n_frames; ++frame) {
                const float value = row[frame];
                if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                    return false;
                }
                mean[frame] += value;
            }
        }
        for (double & value : mean) {
            value /= n_tokens;
        }

        std::fill(inv_std.begin(), inv_std.end(), 0.0);
        for (int token = 0; token < n_tokens; ++token) {
            const float * row = values + static_cast<size_t>(token) * n_frames;
            for (int frame = 0; frame < n_frames; ++frame) {
                const double delta = row[frame] - mean[frame];
                inv_std[frame] += delta * delta;
            }
        }
        for (double & value : inv_std) {
            // Constant columns carry no evidence for any token at this frame.
            value = value > 0.0 ? 1.0 / std::sqrt(value / n_tokens) : 0.0;
        }

        for (int token = 0; token < n_alignment_rows; ++token) {
            const float * row   = values + static_cast<size_t>(first_alignment_row + token) * n_frames;
            float *       score = scores.data() + static_cast<size_t>(token) * n_frames;
            for (int frame = 0; frame < n_frames; ++frame) {
                normalized[frame] = static_cast<float>((row[frame] - mean[frame]) * inv_std[frame]);
            }
            for (int frame = 0; frame < n_frames; ++frame) {
                score[frame] += median_at(normalized, frame);
            }
        }
    }

    dtw_boundaries(scores, n_alignment_rows, n_frames, n_heads, token_frames);
    return true;
}

}  // namespace transcribe::whisper
