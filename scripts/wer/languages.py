#!/usr/bin/env python3
"""languages.py — the canonical BCP-47 code table for the WER tooling.

One table, three consumers: the language names in reports, the `languages`
table in fleurs.db, and the script column used to explain script-fold
scoring (the Traditional-vs-Simplified case on Breeze/zh is a fact about
script, not a free-text note).

It covers every code that appears ANYWHERE: benchmarked FLEURS languages,
FLEURS configs not yet benchmarked, and languages a model card merely
claims. Coverage a card asserts but that nothing has measured is still a
language the site has to be able to name.

Names are checked in as literals rather than resolved from a locale library
at runtime, so the tooling keeps its no-dependency property and the names
stay what a user would call the language. Generated once from CLDR via
langcodes, then the names already in use were kept verbatim so no report
churns. Edit by hand; regeneration is not part of any build.

Scripts are ISO 15924. For the three codes where the script is genuinely
contested, the FLEURS config name settles it (cmn_hans_cn -> Hans,
yue_hant_hk -> Hant); everything else is the CLDR likely-script.
"""
from __future__ import annotations

# code -> (English name, ISO 15924 script)
LANGUAGES: dict[str, tuple[str, str]] = {
    'af': ('Afrikaans', 'Latn'),
    'am': ('Amharic', 'Ethi'),
    'ar': ('Arabic', 'Arab'),
    'as': ('Assamese', 'Beng'),
    'ast': ('Asturian', 'Latn'),
    'az': ('Azerbaijani', 'Latn'),
    'ba': ('Bashkir', 'Cyrl'),
    'be': ('Belarusian', 'Cyrl'),
    'bg': ('Bulgarian', 'Cyrl'),
    'bn': ('Bangla', 'Beng'),
    'bo': ('Tibetan', 'Tibt'),
    'br': ('Breton', 'Latn'),
    'bs': ('Bosnian', 'Latn'),
    'ca': ('Catalan', 'Latn'),
    'ceb': ('Cebuano', 'Latn'),
    'ckb': ('Central Kurdish', 'Arab'),
    'cs': ('Czech', 'Latn'),
    'cy': ('Welsh', 'Latn'),
    'da': ('Danish', 'Latn'),
    'de': ('German', 'Latn'),
    'el': ('Greek', 'Grek'),
    'en': ('English', 'Latn'),
    'es': ('Spanish', 'Latn'),
    'et': ('Estonian', 'Latn'),
    'eu': ('Basque', 'Latn'),
    'fa': ('Persian', 'Arab'),
    'ff': ('Fula', 'Latn'),
    'fi': ('Finnish', 'Latn'),
    'fil': ('Filipino', 'Latn'),
    'fo': ('Faroese', 'Latn'),
    'fr': ('French', 'Latn'),
    'ga': ('Irish', 'Latn'),
    'gl': ('Galician', 'Latn'),
    'gu': ('Gujarati', 'Gujr'),
    'ha': ('Hausa', 'Latn'),
    'haw': ('Hawaiian', 'Latn'),
    'he': ('Hebrew', 'Hebr'),
    'hi': ('Hindi', 'Deva'),
    'hr': ('Croatian', 'Latn'),
    'ht': ('Haitian Creole', 'Latn'),
    'hu': ('Hungarian', 'Latn'),
    'hy': ('Armenian', 'Armn'),
    'id': ('Indonesian', 'Latn'),
    'ig': ('Igbo', 'Latn'),
    'is': ('Icelandic', 'Latn'),
    'it': ('Italian', 'Latn'),
    'ja': ('Japanese', 'Jpan'),
    'jv': ('Javanese', 'Latn'),
    'jw': ('Javanese', 'Latn'),
    'ka': ('Georgian', 'Geor'),
    'kam': ('Kamba', 'Latn'),
    'kea': ('Kabuverdianu', 'Latn'),
    'kk': ('Kazakh', 'Cyrl'),
    'km': ('Khmer', 'Khmr'),
    'kn': ('Kannada', 'Knda'),
    'ko': ('Korean', 'Kore'),
    'ky': ('Kyrgyz', 'Cyrl'),
    'la': ('Latin', 'Latn'),
    'lb': ('Luxembourgish', 'Latn'),
    'lg': ('Ganda', 'Latn'),
    'ln': ('Lingala', 'Latn'),
    'lo': ('Lao', 'Laoo'),
    'lt': ('Lithuanian', 'Latn'),
    'luo': ('Luo (Kenya and Tanzania)', 'Latn'),
    'lv': ('Latvian', 'Latn'),
    'mg': ('Malagasy', 'Latn'),
    'mi': ('Māori', 'Latn'),
    'mk': ('Macedonian', 'Cyrl'),
    'ml': ('Malayalam', 'Mlym'),
    'mn': ('Mongolian', 'Cyrl'),
    'mr': ('Marathi', 'Deva'),
    'ms': ('Malay', 'Latn'),
    'mt': ('Maltese', 'Latn'),
    'my': ('Burmese', 'Mymr'),
    'nb': ('Norwegian Bokmal', 'Latn'),
    'ne': ('Nepali', 'Deva'),
    'nl': ('Dutch', 'Latn'),
    'nn': ('Norwegian Nynorsk', 'Latn'),
    'no': ('Norwegian', 'Latn'),
    'nso': ('Northern Sotho', 'Latn'),
    'ny': ('Nyanja', 'Latn'),
    'oc': ('Occitan', 'Latn'),
    'om': ('Oromo', 'Latn'),
    'or': ('Odia', 'Orya'),
    'pa': ('Punjabi', 'Guru'),
    'pl': ('Polish', 'Latn'),
    'ps': ('Pashto', 'Arab'),
    'pt': ('Portuguese', 'Latn'),
    'ro': ('Romanian', 'Latn'),
    'ru': ('Russian', 'Cyrl'),
    'sa': ('Sanskrit', 'Deva'),
    'sd': ('Sindhi', 'Arab'),
    'si': ('Sinhala', 'Sinh'),
    'sk': ('Slovak', 'Latn'),
    'sl': ('Slovenian', 'Latn'),
    'sn': ('Shona', 'Latn'),
    'so': ('Somali', 'Latn'),
    'sq': ('Albanian', 'Latn'),
    'sr': ('Serbian', 'Cyrl'),
    'su': ('Sundanese', 'Latn'),
    'sv': ('Swedish', 'Latn'),
    'sw': ('Swahili', 'Latn'),
    'ta': ('Tamil', 'Taml'),
    'te': ('Telugu', 'Telu'),
    'tg': ('Tajik', 'Cyrl'),
    'th': ('Thai', 'Thai'),
    'tk': ('Turkmen', 'Latn'),
    'tl': ('Filipino', 'Latn'),
    'tr': ('Turkish', 'Latn'),
    'tt': ('Tatar', 'Cyrl'),
    'uk': ('Ukrainian', 'Cyrl'),
    'umb': ('Umbundu', 'Latn'),
    'ur': ('Urdu', 'Arab'),
    'uz': ('Uzbek', 'Latn'),
    'vi': ('Vietnamese', 'Latn'),
    'wo': ('Wolof', 'Latn'),
    'xh': ('Xhosa', 'Latn'),
    'yi': ('Yiddish', 'Hebr'),
    'yo': ('Yoruba', 'Latn'),
    'yue': ('Cantonese', 'Hant'),
    'zh': ('Mandarin Chinese', 'Hans'),
    'zh-cn': ('Chinese (China)', 'Hans'),
    'zu': ('Zulu', 'Latn'),
}


def name(code: str) -> str:
    """English name, falling back to the code itself for anything unknown."""
    row = LANGUAGES.get(code)
    return row[0] if row else code


def script(code: str) -> str | None:
    row = LANGUAGES.get(code)
    return (row[1] or None) if row else None
