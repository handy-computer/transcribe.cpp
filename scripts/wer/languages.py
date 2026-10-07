"""Lightweight language metadata shared by WER ingestion and catalog policy."""
from __future__ import annotations

# BCP-47 shorthand -> google/fleurs configuration. Keep this module stdlib-only:
# catalog publication planning imports it without installing dataset/audio deps.
FLEURS_LANGS: dict[str, str] = {
    # African
    "af": "af_za", "am": "am_et", "ff": "ff_sn", "ha": "ha_ng",
    "ig": "ig_ng", "kam": "kam_ke", "kea": "kea_cv", "lg": "lg_ug",
    "ln": "ln_cd", "luo": "luo_ke", "nso": "nso_za", "ny": "ny_mw",
    "om": "om_et", "sn": "sn_zw", "so": "so_so", "sw": "sw_ke",
    "umb": "umb_ao", "wo": "wo_sn", "xh": "xh_za", "yo": "yo_ng",
    "zu": "zu_za",
    # Arabic, Hebrew, Persian, Kurdish
    "ar": "ar_eg", "he": "he_il", "fa": "fa_ir", "ckb": "ckb_iq",
    "ps": "ps_af", "ur": "ur_pk",
    # South Asian
    "as": "as_in", "bn": "bn_in", "gu": "gu_in", "hi": "hi_in",
    "kn": "kn_in", "ml": "ml_in", "mr": "mr_in", "ne": "ne_np",
    "or": "or_in", "pa": "pa_in", "sd": "sd_in", "ta": "ta_in",
    "te": "te_in",
    # East / Southeast Asian
    "my": "my_mm", "fil": "fil_ph", "tl": "fil_ph",
    "id": "id_id", "ja": "ja_jp", "jv": "jv_id", "jw": "jv_id",
    "km": "km_kh",
    "ko": "ko_kr", "lo": "lo_la", "ms": "ms_my", "th": "th_th",
    "vi": "vi_vn", "ceb": "ceb_ph",
    # Chinese / Cantonese
    "zh": "cmn_hans_cn", "zh-cn": "cmn_hans_cn",
    # zh-tw is intentionally absent: FLEURS has no Traditional Mandarin.
    "yue": "yue_hant_hk",
    # Central Asian
    "az": "az_az", "kk": "kk_kz", "ky": "ky_kg", "mn": "mn_mn",
    "tg": "tg_tj", "uz": "uz_uz", "hy": "hy_am", "ka": "ka_ge",
    # European
    "ast": "ast_es", "be": "be_by", "bg": "bg_bg", "bs": "bs_ba",
    "ca": "ca_es", "cs": "cs_cz", "cy": "cy_gb", "da": "da_dk",
    "de": "de_de", "el": "el_gr", "en": "en_us", "es": "es_419",
    "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr", "ga": "ga_ie",
    "gl": "gl_es", "hr": "hr_hr", "hu": "hu_hu", "is": "is_is",
    "it": "it_it", "lb": "lb_lu", "lt": "lt_lt", "lv": "lv_lv",
    "mi": "mi_nz", "mk": "mk_mk", "mt": "mt_mt", "nb": "nb_no",
    "no": "nb_no", "nl": "nl_nl", "oc": "oc_fr", "pl": "pl_pl",
    "pt": "pt_br", "ro": "ro_ro", "ru": "ru_ru", "sk": "sk_sk",
    "sl": "sl_si", "sr": "sr_rs", "sv": "sv_se", "tr": "tr_tr",
    "uk": "uk_ua",
}

# First spelling for a configuration is the canonical catalog spelling. This
# makes aliases deterministic: jw -> jv, tl -> fil, no -> nb, zh-cn -> zh.
FLEURS_CANONICAL_BY_CONFIG: dict[str, str] = {}
for _language, _config in FLEURS_LANGS.items():
    FLEURS_CANONICAL_BY_CONFIG.setdefault(_config, _language)

# Character-based scoring for scripts without reliable whitespace-delimited
# words in FLEURS. Khmer/Lao/Burmese spaces are phrase separators rather than
# dependable word boundaries, so WER would mostly measure orthography policy.
CER_LANGUAGES = {"zh", "yue", "ja", "ko", "th", "km", "lo", "my"}

# Model-side spellings that name the same language as a dataset code. A
# language hint is checked against the manifest through this map, so a model
# that prompts with Whisper's legacy `jw` can be scored on FLEURS `jv`.
LANGUAGE_ALIASES = {"jw": "jv", "tl": "fil", "no": "nb", "zh-cn": "zh"}
