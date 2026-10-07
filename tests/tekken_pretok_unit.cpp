// tekken_pretok_unit.cpp - pure-host test of the Tekken (Mistral /
// Voxtral) pretokenizer split, no model required.
//
// Expected splits are regex.findall() with the tekken.json pattern,
// except the ASCII-only case cases marked below.

#include "transcribe-unicode.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

std::string byte_encode(const std::string & raw) {
    std::string out;
    for (char c : raw) {
        out += transcribe::unicode::byte_to_unicode(static_cast<uint8_t>(c));
    }
    return out;
}

struct Case {
    const char *             text;
    std::vector<std::string> expected;  // raw UTF-8 pretokens
};

void check_case(const Case & c) {
    const std::vector<std::string> got = transcribe::unicode::pretokenize_tekken(c.text);
    bool                           ok  = got.size() == c.expected.size();
    for (size_t i = 0; ok && i < got.size(); ++i) {
        ok = got[i] == byte_encode(c.expected[i]);
    }
    if (!ok) {
        std::fprintf(stderr, "FAIL pretokenize_tekken(\"%s\"): got %zu pieces, expected %zu\n", c.text, got.size(),
                     c.expected.size());
        ++g_failures;
    }
}

}  // namespace

int main() {
    const std::vector<Case> cases = {
        // Case split: lower->Upper boundary splits, UPPER+lower stays one word.
        { u8"iPhone McDonald HTTPServer",                                                { u8"i", u8"Phone", u8" Mc", u8"Donald", u8" HTTPServer" } },
        { u8"ABC aBC AbC",                                                               { u8"ABC", u8" a", u8"BC", u8" Ab", u8"C" }                },
        // No contraction alternative: the apostrophe prefixes the letters.
        { u8"don't I'M",                                                                 { u8"don", u8"'t", u8" I", u8"'M" }                        },
        // Symbol runs swallow a trailing [\r\n/]*.
        { u8"!\n/x",                                                                     { u8"!\n/", u8"x" }                                        },
        { u8"a /\r\n/ b",                                                                { u8"a", u8" /\r\n/", u8" b" }                             },
        // ASCII-only case: non-ASCII letters sit in both classes, so the
        // regex split before U+00C4 does not happen.
        { u8"\u00d6l\u00c4nderung \u01c5emal",                                           { u8"\u00d6l\u00c4nderung", u8" \u01c5emal" }              },
        // Lm (U+02B0) sits in both classes.
        { u8"a\u02b0B",                                                                  { u8"a\u02b0", u8"B" }                                     },
        // Combining marks join letter runs, or stand alone after a prefix.
        { u8"e\u0301cole \u0301 !\u0301a",                                               { u8"e\u0301cole", u8" \u0301", u8" !\u0301", u8"a" }      },
        // Devanagari: vowel signs / virama are \p{M}, so words stay whole.
        { u8"\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
         { u8"\u0928\u092e\u0938\u094d\u0924\u0947", u8" \u0926\u0941\u0928\u093f\u092f\u093e" }                                                    },
        // Single-codepoint \p{N}, including non-ASCII digits / fractions.
        { u8"12 \u0663\u00bd",                                                           { u8"1", u8"2", u8" ", u8"\u0663", u8"\u00bd" }            },
        // Whitespace alternatives.
        { u8"  x  \n\n  y   ",                                                           { u8" ", u8" x", u8"  \n\n", u8" ", u8" y", u8"   " }      },
        // Supplementary-plane letters (binary-search path of the flags table).
        { u8"\U00010400\U00010428 \U0001D400",                                           { u8"\U00010400\U00010428", u8" \U0001D400" }              },
        { u8"",                                                                          {}                                                         },
    };
    for (const Case & c : cases) {
        check_case(c);
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "tekken_pretok_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "tekken_pretok_unit: ok\n");
    return EXIT_SUCCESS;
}
