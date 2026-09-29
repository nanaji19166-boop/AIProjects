/**
 * indic_processor.js — JS port of AI4Bharat IndicTransToolkit's IndicProcessor.
 *
 * Two-stage wrapper around the IndicTrans2 model:
 *   - preprocessBatch(batch, srcLang, tgtLang) → strings prefixed with FLORES tags,
 *     punctuation-normalized, MosesTokenized (English source), with URLs / numerals
 *     / emails / @-handles replaced by <ID{n}> placeholders. Placeholder maps
 *     are pushed onto a per-instance FIFO queue.
 *   - postprocessBatch(sents, lang) → drains the queue, fixes Perso-Arabic /
 *     Oriya scripts, restores placeholders, transliterates from Devanagari to
 *     the target script (the model's output is Devanagari-normalized for
 *     non-Devanagari Indic languages), and trivial-detokenizes.
 *
 * Ported from IndicTransToolkit/processor.pyx (538 LOC) plus the four functions
 * the Python class actually reaches in indic_nlp_library:
 *   - UnicodeIndicTransliterator.transliterate (offset arithmetic on
 *     SCRIPT_RANGES, with a Tamil-specific consonant collapse).
 *   - indic_detokenize.trivial_detokenize (regex-based attach-left/right rules
 *     + alternating-quote pairing).
 *   - sacremoses MosesPunctNormalizer + MosesTokenizer + MosesDetokenizer for
 *     the English source side. We ship a minimal, fixture-validated subset —
 *     full sacremoses parity isn't necessary because SentencePiece (downstream)
 *     handles the real tokenization; Moses just splits punctuation off so the
 *     SPM merges line up with the AI4Bharat training distribution.
 *
 * Validated against fixtures/expected_processed.json (528 records over 11
 * Indic languages × 4 fixture categories). Run js/indic_processor.test.js
 * via Node to verify parity.
 */

// ---------------------------------------------------------------------------
// Constants — data tables
// ---------------------------------------------------------------------------

// FLORES code → ISO 639-1 (or pseudo-code for trans-script aliases).
// Verbatim from processor.pyx:59-94.
const FLORES_TO_ISO = {
  asm_Beng: "as", awa_Deva: "hi", ben_Beng: "bn", bho_Deva: "hi",
  brx_Deva: "hi", doi_Deva: "hi", eng_Latn: "en", gom_Deva: "kK",
  gon_Deva: "hi", guj_Gujr: "gu", hin_Deva: "hi", hne_Deva: "hi",
  kan_Knda: "kn", kas_Arab: "ur", kas_Deva: "hi", kha_Latn: "en",
  lus_Latn: "en", mag_Deva: "hi", mai_Deva: "hi", mal_Mlym: "ml",
  mar_Deva: "mr", mni_Beng: "bn", mni_Mtei: "hi", npi_Deva: "ne",
  ory_Orya: "or", pan_Guru: "pa", san_Deva: "hi", sat_Olck: "or",
  snd_Arab: "ur", snd_Deva: "hi", tam_Taml: "ta", tel_Telu: "te",
  urd_Arab: "ur", unr_Deva: "hi",
};

// ISO → [base codepoint, top codepoint]. From indicnlp/langinfo.py:14-29.
const SCRIPT_RANGES = {
  pa: [0x0a00, 0x0a7f],
  gu: [0x0a80, 0x0aff],
  or: [0x0b00, 0x0b7f],
  ta: [0x0b80, 0x0bff],
  te: [0x0c00, 0x0c7f],
  kn: [0x0c80, 0x0cff],
  ml: [0x0d00, 0x0d7f],
  si: [0x0d80, 0x0dff],
  hi: [0x0900, 0x097f],
  mr: [0x0900, 0x097f],
  kK: [0x0900, 0x097f],
  sa: [0x0900, 0x097f],
  ne: [0x0900, 0x097f],
  sd: [0x0900, 0x097f],
  bn: [0x0980, 0x09ff],
  as: [0x0980, 0x09ff],
};

const COORDINATED_RANGE_START = 0x00;
const COORDINATED_RANGE_END = 0x6f;

// Indic digit codepoint → ASCII. From processor.pyx:99-140.
const INDIC_DIGIT_MAP = {
  // 0
  "০": "0", "૦": "0", "೦": "0", "०": "0",
  "٠": "0", "꯰": "0", "୦": "0", "੦": "0",
  "᱐": "0", "۰": "0",
  // 1
  "১": "1", "૧": "1", "१": "1", "೧": "1",
  "۱": "1", "꯱": "1", "୧": "1", "੧": "1",
  "᱑": "1", "౧": "1",
  // 2
  "২": "2", "૨": "2", "२": "2", "೨": "2",
  "۲": "2", "꯲": "2", "୨": "2", "੨": "2",
  "᱒": "2", "౨": "2",
  // 3
  "৩": "3", "૩": "3", "३": "3", "೩": "3",
  "۳": "3", "꯳": "3", "୩": "3", "੩": "3",
  "᱓": "3", "౩": "3",
  // 4
  "৪": "4", "૪": "4", "४": "4", "೪": "4",
  "۴": "4", "꯴": "4", "୪": "4", "੪": "4",
  "᱔": "4", "౪": "4",
  // 5
  "৫": "5", "૫": "5", "५": "5", "೫": "5",
  "۵": "5", "꯵": "5", "୫": "5", "੫": "5",
  "᱕": "5", "౫": "5",
  // 6
  "৬": "6", "૬": "6", "६": "6", "೬": "6",
  "۶": "6", "꯶": "6", "୬": "6", "੬": "6",
  "᱖": "6", "౬": "6",
  // 7
  "৭": "7", "૭": "7", "७": "7", "೭": "7",
  "۷": "7", "꯷": "7", "୭": "7", "੭": "7",
  "᱗": "7", "౭": "7",
  // 8
  "৮": "8", "૮": "8", "८": "8", "೮": "8",
  "۸": "8", "꯸": "8", "୮": "8", "੮": "8",
  "᱘": "8", "౮": "8",
  // 9
  "৯": "9", "૯": "9", "९": "9", "೯": "9",
  "۹": "9", "꯹": "9", "୯": "9", "੯": "9",
  "᱙": "9", "౯": "9",
};

// Indic failure-case strings for placeholder mapping.
// Verbatim from processor.pyx _INDIC_FAILURE_CASES.
const INDIC_FAILURE_CASES = [
  "آی ڈی ", "ꯑꯥꯏꯗꯤ", "आईडी", "आई . डी . ", "आई . डी .",
  "आई. डी. ", "आई. डी.", "आय. डी. ", "आय. डी.",
  "आय . डी . ", "आय . डी .आइ . डी . ", "आइ . डी .",
  "आइ. डी. ", "आइ. डी.", "ऐटि", "آئی ڈی ", "ᱟᱭᱰᱤ ᱾",
  "आयडी", "ऐडि", "आइडि", "ᱟᱭᱰᱤ",
];

// Punctuation replacements applied in order. From processor.pyx:_PUNC_REPLACEMENTS.
// Mix of regex.replace + literal mapping.
const PUNC_REPLACEMENTS = [
  [/\r/g, ""],
  [/\(\s*/g, "("],
  [/\s*\)/g, ")"],
  [/\s:\s?/g, ":"],
  [/\s;\s?/g, ";"],
  [/[`´‘‚’]/g, "'"],
  [/[„“”«»]/g, '"'],
  [/[–—]/g, "-"],
  [/\.\.\./g, "..."],
  [/ %/g, "%"],
  [/nº /g, "nº "],
  [/ ºC/g, " ºC"],
  [/ [?!;]/g, (m) => m.trim()],
  [/, /g, ", "],
];

// Patterns for placeholder masking. Order matters: EMAIL → URL → NUMERAL → OTHER.
// Verbatim from processor.pyx _URL_PATTERN / _NUMERAL_PATTERN / _EMAIL_PATTERN /
// _OTHER_PATTERN. Note the URL pattern uses a lookbehind — V8 / WebKit support
// this since 2018 / 2020 respectively.
const PATTERNS = {
  email: /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}/g,
  url: /\b(?<![\w/.])(?:(?:https?|ftp):\/\/)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b/g,
  numeral: /(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-/.,:']\d+[-/.,:'+]\d+(?:\.\d+)?|\d+[-/.:'+]\d+(?:\.\d+)?)/g,
  other: /[A-Za-z0-9]*[#|@]\w+/g,
};

// Cleanup regexes for _punc_norm.
const MULTISPACE_REGEX = /[ ]{2,}/g;
const END_BRACKET_SPACE_PUNC_REGEX = /\) ([.!:?;,])/g;
const DIGIT_SPACE_PERCENT = /(\d) %/g;
const DOUBLE_QUOT_PUNC = /"([,.]+)/g;
const DIGIT_NBSP_DIGIT = /(\d) (\d)/g;


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Apply PUNC_REPLACEMENTS in sequence + cleanup regexes. Verbatim _punc_norm. */
function puncNorm(text) {
  for (const [pat, repl] of PUNC_REPLACEMENTS) {
    text = text.replace(pat, repl);
  }
  text = text.replace(MULTISPACE_REGEX, " ");
  text = text.replace(END_BRACKET_SPACE_PUNC_REGEX, ")$1");
  text = text.replace(DIGIT_SPACE_PERCENT, "$1%");
  text = text.replace(DOUBLE_QUOT_PUNC, '$1"');
  text = text.replace(DIGIT_NBSP_DIGIT, "$1.$2");
  return text.trim();
}

/** Translate Indic digits to ASCII via INDIC_DIGIT_MAP. */
function normalizeDigits(text) {
  let out = "";
  for (const ch of text) {
    out += INDIC_DIGIT_MAP[ch] ?? ch;
  }
  return out;
}

/**
 * Wrap matched URL/EMAIL/NUMERAL/OTHER substrings with <ID{n}> placeholders.
 * Returns [maskedText, placeholderMap]. The map contains MANY variants for
 * each match (case, brackets, Indic-failure-prefix combos) so postprocessing
 * can recover even if the SentencePiece re-encoding mangles the placeholder.
 */
function wrapWithPlaceholders(text) {
  const map = {};
  let serialNo = 1;
  // Order: email → url → numeral → other (matches processor.pyx).
  const orderedPatterns = [PATTERNS.email, PATTERNS.url, PATTERNS.numeral, PATTERNS.other];

  for (const pattern of orderedPatterns) {
    // Reset lastIndex on global regex; collect unique matches.
    pattern.lastIndex = 0;
    const matches = new Set();
    let m;
    while ((m = pattern.exec(text)) !== null) {
      matches.add(m[0]);
      if (m.index === pattern.lastIndex) pattern.lastIndex++;
    }

    for (const match of matches) {
      // Length filters from processor.pyx:283-289.
      if (pattern === PATTERNS.url) {
        if (match.replace(/\./g, "").length < 4) continue;
      }
      if (pattern === PATTERNS.numeral) {
        if (match.replace(/ /g, "").replace(/\./g, "").replace(/:/g, "").length < 4) continue;
      }

      const base = `<ID${serialNo}>`;

      // Variants the postprocessor might encounter (model + tokenizer can mangle).
      const variants = [
        `<ID${serialNo}>`, `< ID${serialNo} >`, `[ID${serialNo}]`,
        `[ ID${serialNo} ]`, `[ID ${serialNo}]`, `<ID${serialNo}]`,
        `< ID${serialNo}]`, `<ID${serialNo} ]`,
        `<id${serialNo}>`, `< id${serialNo} >`, `[id${serialNo}]`,
        `[ id${serialNo} ]`, `[id ${serialNo}]`, `<id${serialNo}]`,
        `< id${serialNo}]`, `<id${serialNo} ]`,
      ];
      for (const v of variants) map[v] = match;

      // Indic-failure-case combinations (processor.pyx:316-328).
      for (const fc of INDIC_FAILURE_CASES) {
        const combos = [
          `<${fc}${serialNo}>`, `< ${fc}${serialNo} >`, `< ${fc} ${serialNo} >`,
          `<${fc} ${serialNo}]`, `< ${fc} ${serialNo} ]`,
          `[${fc}${serialNo}]`, `[${fc} ${serialNo}]`,
          `[ ${fc}${serialNo} ]`, `[ ${fc} ${serialNo} ]`,
          `${fc} ${serialNo}`, `${fc}${serialNo}`,
        ];
        for (const c of combos) map[c] = match;
      }

      // Replace ALL occurrences of `match` with `base` (literal, not regex).
      text = text.split(match).join(base);
      serialNo += 1;
    }
  }

  // Cleanup mirroring processor.pyx:332-333.
  text = text.replace(/\s+/g, " ").replace(/>\//g, ">").replace(/]\//g, "]");
  return [text, map];
}

/** Replace each placeholder variant in text with its original substring. */
function restorePlaceholders(text, map) {
  // Order: longer keys first so e.g. "<ID10>" is restored before "<ID1>".
  // Then exact-match string replace (Python's .replace is literal).
  const keys = Object.keys(map).sort((a, b) => b.length - a.length);
  for (const k of keys) {
    text = text.split(k).join(map[k]);
  }
  return text;
}

/**
 * Minimal MosesPunctNormalizer port. Applies Unicode-quote / dash / ellipsis
 * normalization that sacremoses's MosesPunctNormalizer does for English.
 * Most of these are also handled by puncNorm; we keep this minimal because
 * IndicProcessor calls _punc_norm BEFORE passing to MosesPunctNormalizer, so
 * the input is already mostly clean.
 */
function mosesPunctNormalize(text) {
  return text
    .replace(/\r/g, "")
    .replace(/\(/g, " (")
    .replace(/\)/g, ") ")
    .replace(/ +/g, " ")
    .replace(/\) ([.!:?;,])/g, ")$1")
    .replace(/\( /g, "(")
    .replace(/ \)/g, ")")
    .replace(/(\d) %/g, "$1%")
    .replace(/ :/g, ":")
    .replace(/ ;/g, ";")
    .replace(/`/g, "'")
    .replace(/''/g, '"')
    .trim();
}

/**
 * Minimal MosesTokenizer for English. Splits punctuation, English contractions
 * (`'s`, `'re`, `'ll`, `'d`, `'m`, `'t`, `'ve`), and percent / currency signs
 * off into separate space-delimited tokens. The SentencePiece tokenizer
 * downstream handles the actual subword segmentation; this just inserts the
 * spaces the AI4Bharat training distribution expects.
 *
 * Deliberately does NOT replicate sacremoses's protected-pattern logic
 * (URLs, dates) — those get masked as <ID{n}> placeholders before we get
 * here, so they're already single tokens.
 */
function mosesTokenize(text) {
  // 1. Pad universal punctuation. Period is special-cased below because
  //    sacremoses preserves decimal numbers (`7.2` stays `7.2`, not `7 . 2`).
  text = text.replace(/([,!?;:()\[\]{}"%₹$€£¥<>=+|@#])/g, " $1 ");

  // 2. Period: protect decimal points (digit.digit) before splitting.
  text = text.replace(/(\d)\.(\d)/g, "$1@@DEC@@$2");
  text = text.replace(/\./g, " . ");
  text = text.replace(/@@DEC@@/g, ".");

  // 3. English contractions: split common 'suffix forms. Sacremoses uses
  //    \b(?=...) for these.
  text = text.replace(/(\w)('s|'re|'ll|'d|'m|'t|'ve)\b/g, "$1 $2");

  // 4. Collapse multispaces, trim.
  return text.replace(/\s+/g, " ").trim();
}

/**
 * Tamil-specific consonant-mapping correction. Tamil lacks voiced & aspirated
 * plosives so we collapse the corresponding Devanagari range. Verbatim from
 * indic_nlp_library/.../unicode_transliterate.py:_correct_tamil_mapping.
 */
function correctTamilMapping(offset) {
  // Consonant rows 1-4 (varnamala): collapse voiced/aspirated to unvoiced
  // unaspirated. Exception: ja (offset 0x1c) has a real Tamil mapping.
  if (
    offset >= 0x15 && offset <= 0x28 &&
    offset !== 0x1c &&
    !((offset - 0x15) % 5 === 0 || (offset - 0x15) % 5 === 4)
  ) {
    const subst = Math.floor((offset - 0x15) / 5);
    offset = 0x15 + 5 * subst;
  }
  // Consonant row 5: collapse to pa.
  if (offset === 0x2b || offset === 0x2c || offset === 0x2d) {
    offset = 0x2a;
  }
  // sh → Sh
  if (offset === 0x36) {
    offset = 0x37;
  }
  return offset;
}

/**
 * Transliterate text from one Brahmi-derived script to another via offset
 * arithmetic on SCRIPT_RANGES. From unicode_transliterate.py:transliterate.
 */
function transliterate(text, fromIso, toIso) {
  if (fromIso === toIso) return text;
  const fromR = SCRIPT_RANGES[fromIso];
  const toR = SCRIPT_RANGES[toIso];
  if (!fromR || !toR) return text;

  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    let offset = cp - fromR[0];
    // Skip Devanagari danda / double-danda even if in the coordinated range.
    if (
      offset >= COORDINATED_RANGE_START &&
      offset <= COORDINATED_RANGE_END &&
      ch !== "।" &&
      ch !== "॥"
    ) {
      if (toIso === "ta") offset = correctTamilMapping(offset);
      out += String.fromCodePoint(toR[0] + offset);
    } else {
      out += ch;
    }
  }
  return out;
}

/**
 * Indic detokenizer. Verbatim port of indic_detokenize.trivial_detokenize_indic:
 *   1. Glue number sequences ("1 , 2" → "1,2") via pat_num_seq
 *   2. Attach left+right punctuation: "-/\\"
 *   3. Attach left punctuation: "!%)\]},.:;>?।॥"
 *   4. Attach right punctuation: "#$(\[{<@"
 *   5. Alternate quote pairing for ', ", ` (right-attach first occurrence,
 *      left-attach second, etc.)
 */
function trivialDetokenize(text /*, iso */) {
  // 1. Number sequences
  const patNumSeq = /([0-9]+ [,.:/] )+[0-9]+/g;
  text = text.replace(patNumSeq, (m) => m.replace(/ /g, ""));

  // 2-4. Attach rules
  text = text.replace(/ ([\-/\\]) /g, "$1");          // lr_attach
  text = text.replace(/ ([!%)\]},.:;>?।॥])/g, "$1"); // left_attach
  text = text.replace(/([#$(\[{<@]) /g, "$1");        // right_attach

  // 5. Alternating quote pairing
  for (const punc of ["'", '"', "`"]) {
    let cnt = 0;
    let buf = "";
    for (const c of text) {
      if (c === punc) {
        buf += cnt % 2 === 0 ? "@RA" : "@LA";
        cnt += 1;
      } else {
        buf += c;
      }
    }
    text = buf
      .split(`@RA `).join(punc)
      .split(` @LA`).join(punc)
      .split("@RA").join(punc)
      .split("@LA").join(punc);
  }
  return text;
}


// ---------------------------------------------------------------------------
// IndicProcessor class
// ---------------------------------------------------------------------------

class IndicProcessor {
  constructor({ inference = true } = {}) {
    this.inference = inference;
    /** FIFO queue of placeholder maps, populated by preprocess, drained by postprocess. */
    this._queue = [];
  }

  // Reset per-call queue (single-translation contract).
  resetQueue() { this._queue = []; }

  _preprocess(sent, srcLang, tgtLang, isTarget) {
    const isoLang = FLORES_TO_ISO[srcLang] ?? "hi";
    const scriptPart = srcLang.split("_")[1];
    const doTransliterate = !["Arab", "Aran", "Olck", "Mtei", "Latn"].includes(scriptPart);

    // 1. Punctuation normalization
    sent = puncNorm(sent);

    // 2. Numerals + (if inference) placeholder masking
    sent = normalizeDigits(sent);
    if (this.inference) {
      const [masked, map] = wrapWithPlaceholders(sent);
      sent = masked;
      this._queue.push(map);
    }

    // 3. English vs Indic path
    let processed;
    if (isoLang === "en") {
      const stripped = sent.trim();
      const normed = mosesPunctNormalize(stripped);
      processed = mosesTokenize(normed);
    } else {
      // Indic source path (we don't fully use this for en→indic, but keep for
      // x→en direction). Without porting per-script normalizers, just trivial
      // tokenize; downstream SentencePiece handles real subword segmentation.
      processed = sent.trim();
      if (doTransliterate) {
        processed = transliterate(processed, isoLang, "hi").replace(/ ् /g, "्");
      }
    }

    processed = processed.trim();
    return isTarget ? processed : `${srcLang} ${tgtLang} ${processed}`;
  }

  _postprocess(sent, lang, providedMap = null) {
    if (Array.isArray(sent)) sent = sent[0];

    const map = providedMap ?? this._queue.shift() ?? {};
    const [langCode, scriptCode] = lang.split("_", 2);
    const isoLang = FLORES_TO_ISO[lang] ?? "hi";

    // Perso-Arabic fixups
    if (scriptCode === "Arab" || scriptCode === "Aran") {
      sent = sent
        .replace(/ ؟/g, "؟")
        .replace(/ ۔/g, "۔")
        .replace(/ ،/g, "،")
        .replace(/ٮ۪/g, "ؠ");
    }

    // Oriya fix
    if (langCode === "ory") {
      sent = sent.replace(/ଯ଼/g, "ୟ");
    }

    // Restore placeholders
    sent = restorePlaceholders(sent, map);

    // Detokenize
    if (lang === "eng_Latn") {
      // English target: defer to a minimal Moses-detok approximation.
      return _mosesDetokenizeEnglish(sent.split(" "));
    }
    const xlated = transliterate(sent, "hi", isoLang);
    return trivialDetokenize(xlated, isoLang);
  }

  preprocessBatch(batch, { srcLang, tgtLang = null, isTarget = false } = {}) {
    if (typeof batch === "string") batch = [batch];
    return batch.map((s) => this._preprocess(s, srcLang, tgtLang, isTarget));
  }

  postprocessBatch(sents, { lang = "hin_Deva", numReturnSequences = 1 } = {}) {
    if (typeof sents === "string") sents = [sents];

    // Drain N=ceil(len(sents)/numReturnSequences) maps off the queue first.
    const numInputs = Math.floor(sents.length / numReturnSequences);
    const maps = [];
    for (let i = 0; i < numInputs; i++) {
      maps.push(this._queue.shift() ?? {});
    }

    const out = [];
    for (let i = 0; i < sents.length; i++) {
      const mapIdx = Math.floor(i / numReturnSequences);
      out.push(this._postprocess(sents[i], lang, maps[mapIdx] ?? {}));
    }
    // Mirror the upstream `self._placeholder_entity_maps.queue.clear()`
    this._queue = [];
    return out;
  }
}


// ---------------------------------------------------------------------------
// Minimal MosesDetokenizer for English target side. Mirror of trivialDetokenize
// for English (attach-left/right rules + alternating quotes).
// ---------------------------------------------------------------------------

function _mosesDetokenizeEnglish(tokens) {
  let text = tokens.join(" ");
  // Attach left for terminal/sentence punctuation
  text = text.replace(/ ([.,!?;:%)\]}>])/g, "$1");
  // Attach right for opening brackets
  text = text.replace(/([(\[{<]) /g, "$1");
  // English contractions: word ' suffix → word'suffix
  text = text.replace(/(\w) ('s|'re|'ll|'d|'m|'t|'ve)\b/g, "$1$2");
  // Alternating quotes
  for (const punc of ['"', "'"]) {
    let cnt = 0;
    let buf = "";
    for (const c of text) {
      if (c === punc) {
        buf += cnt % 2 === 0 ? "@RA" : "@LA";
        cnt += 1;
      } else {
        buf += c;
      }
    }
    text = buf
      .split(`@RA `).join(punc)
      .split(` @LA`).join(punc)
      .split("@RA").join(punc)
      .split("@LA").join(punc);
  }
  return text.replace(/\s+/g, " ").trim();
}


// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

    /* ========================================================================
     * tokenizer — inlined from browser-prep/js/tokenizer.js
     * ====================================================================== */
/**
 * tokenizer.js — minimal BPE tokenizer wrapper for the IT2 ONNX bundle.
 *
 * Loads the two tokenizer.json files (src + tgt) plus tokenizer_meta.json from
 * an HF repo (or local path) and exposes:
 *   - encodeSrc(text)    → Int32Array of input_ids (with extended-vocab IDs
 *                          remapped to <unk>=3 so the encoder ONNX accepts them).
 *   - decodeTgt(ids)     → string (skip_special_tokens=True, no
 *                          clean_up_tokenization_spaces — matches the captured
 *                          Python truth from `clean_up_tokenization_spaces=False`).
 *
 * Why a minimal port instead of @huggingface/transformers's AutoTokenizer:
 *   1. AutoTokenizer assumes a single tokenizer.json; we have two.
 *   2. Our tokenizers need the `>= dict_size → 3` remap on the encoder side
 *      (see browser-prep/scripts/02_build_tokenizer.py). Wrapping AutoTokenizer
 *      to inject that remap is more code than just running BPE here.
 *
 * Implements BPE encoding from `model.vocab` + `model.merges` per the HF
 * tokenizers JSON spec, plus the four pipeline pieces our exporter wrote:
 *   normalizer  → NFKC + collapse repeated whitespace
 *   pre_tokenizer → Metaspace ('▁' replacement, prepend always)
 *   model       → BPE
 *   post_processor → TemplateProcessing  ($A </s>)
 *   decoder     → Metaspace
 * AddedTokens (special tokens + 33 language tags) match before BPE runs.
 *
 * Validated by tokenizer.test.js — encodes the same ID sequence as the Python
 * fast tokenizer for our 15 SRC + 3 TGT parity fixtures.
 */

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/**
 * Build a tokenizer pair from a base URL (local or HF). Fetches:
 *   tokenizer_src.json, tokenizer_tgt.json, tokenizer_meta.json
 */
async function loadIT2Tokenizer(baseUrl) {
  const trim = baseUrl.replace(/\/+$/, "");
  const [src, tgt, meta] = await Promise.all([
    fetch(`${trim}/tokenizer_src.json`).then((r) => r.json()),
    fetch(`${trim}/tokenizer_tgt.json`).then((r) => r.json()),
    fetch(`${trim}/tokenizer_meta.json`).then((r) => r.json()),
  ]);
  return new IT2Tokenizer(src, tgt, meta);
}

// ---------------------------------------------------------------------------
// IT2Tokenizer
// ---------------------------------------------------------------------------

class IT2Tokenizer {
  constructor(srcJson, tgtJson, meta) {
    this.src = new BPETokenizer(srcJson);
    this.tgt = new BPETokenizer(tgtJson);
    this.meta = meta;          // { src_dict_size, tgt_dict_size, unk_id }
  }

  /**
   * Encode a preprocessed source string (already prefixed with FLORES tags
   * like "eng_Latn hin_Deva …"). Returns input_ids + attention_mask as
   * regular arrays, with extended-vocab IDs remapped to <unk>.
   */
  encodeSrc(text) {
    const ids = this.src.encode(text);
    const dictSize = this.meta.src_dict_size;
    const unk = this.meta.unk_id;
    const safe = ids.map((i) => (i < dictSize ? i : unk));
    return {
      input_ids: safe,
      attention_mask: new Array(safe.length).fill(1),
    };
  }

  /** Decode a sequence of decoder output IDs. Drops specials, no cleanup. */
  decodeTgt(ids) {
    const dictSize = this.meta.tgt_dict_size;
    const unk = this.meta.unk_id;
    const safe = ids.map((i) => (i < dictSize ? i : unk));
    return this.tgt.decode(safe, { skip_special_tokens: true });
  }
}

// ---------------------------------------------------------------------------
// BPETokenizer — a vanilla port of just the tokenizer.json features we use.
// ---------------------------------------------------------------------------

class BPETokenizer {
  constructor(json) {
    this.json = json;
    const m = json.model;
    if (m.type !== "BPE") throw new Error(`Expected BPE model, got ${m.type}`);

    // {token: id} from tokenizer.json. Used both for encoding (pieces → ids)
    // and decoding (id → piece via reverse map below).
    this.vocab = m.vocab;
    this.idToToken = new Array(this._maxId() + 1);
    for (const [tok, id] of Object.entries(this.vocab)) {
      this.idToToken[id] = tok;
    }

    // Merges: array of [left, right] pairs OR space-separated strings.
    // Lower index = earlier in BPE training = higher priority.
    this.mergeRanks = new Map();
    if (Array.isArray(m.merges)) {
      m.merges.forEach((pair, idx) => {
        const key = Array.isArray(pair) ? pair.join(" ") : pair;
        this.mergeRanks.set(key, idx);
      });
    }

    // AddedTokens — match in input before BPE runs. Sort by length desc so
    // longer matches win (e.g. `eng_Latn` before any sub-token).
    this.addedTokens = (json.added_tokens || []).slice()
      .sort((a, b) => b.content.length - a.content.length);
    this.addedTokenIds = new Set(this.addedTokens.map((t) => t.id));
    this.specialIds = new Set(
      this.addedTokens.filter((t) => t.special).map((t) => t.id),
    );

    // Normalizer — we only support Sequence([NFKC, Replace(" {2,}", " ")])
    // which is what our 02_build_tokenizer.py produces.
    this.normalizer = json.normalizer;

    // Pre-tokenizer + decoder — we only support Metaspace.
    this.preTokenizer = json.pre_tokenizer;
    this.decoderSpec  = json.decoder;
    this.replacement  = (this.preTokenizer && this.preTokenizer.replacement) || "▁";

    // Post-processor — we support TemplateProcessing single="$A </s>".
    this.postProcessor = json.post_processor;
  }

  _maxId() {
    let max = 0;
    for (const id of Object.values(this.vocab)) if (id > max) max = id;
    for (const t of this.json.added_tokens || []) if (t.id > max) max = t.id;
    return max;
  }

  /** Pipeline: normalize → addedToken split → metaspace pre-tok → BPE → post-proc. */
  encode(text) {
    text = this._normalize(text);
    const chunks = this._splitOnAddedTokens(text);

    const ids = [];
    for (const chunk of chunks) {
      if (typeof chunk === "number") {
        ids.push(chunk);                  // an AddedToken id
        continue;
      }
      // Metaspace pre-tokenize: replace spaces with ▁, prepend ▁ to start.
      let pre = chunk.replace(/ /g, this.replacement);
      if (this.preTokenizer && this.preTokenizer.prepend_scheme === "always") {
        if (!pre.startsWith(this.replacement)) pre = this.replacement + pre;
      }
      // Pre-tokenizer in HF outputs one "word" per consecutive non-space run;
      // for Metaspace with prepend_scheme=always, the entire chunk is one word.
      // Apply BPE to each word separately:
      const words = this._splitWords(pre);
      for (const word of words) {
        for (const id of this._bpeEncode(word)) ids.push(id);
      }
    }

    return this._applyPostProcessor(ids);
  }

  _normalize(text) {
    if (!this.normalizer) return text;
    const apply = (norm) => {
      if (norm.type === "Sequence") {
        for (const sub of norm.normalizers) text = apply(sub) || text;
        return text;
      }
      if (norm.type === "NFKC") return text.normalize("NFKC");
      if (norm.type === "Replace") {
        const pat = norm.pattern.Regex || norm.pattern.String || "";
        const flags = norm.pattern.Regex ? "g" : "";
        return text.replace(new RegExp(pat, flags), norm.content);
      }
      // Fallback: pass through
      return text;
    };
    return apply(this.normalizer);
  }

  /** Split text into [string, addedTokenId, string, …] respecting lstrip/rstrip. */
  _splitOnAddedTokens(text) {
    if (!this.addedTokens.length) return [text];
    // Build a regex matching any added token; prefer special tokens.
    const escaped = this.addedTokens.map((t) =>
      t.content.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    );
    const re = new RegExp(`(${escaped.join("|")})`, "g");
    const out = [];
    let pos = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      const tok = this.addedTokens.find((t) => t.content === m[1]);
      let pre  = text.slice(pos, m.index);
      let post = "";
      if (tok.lstrip) pre = pre.replace(/[ \t]+$/, "");
      // We'll handle rstrip on the next pre by trimming its leading space.
      if (pre.length) out.push(pre);
      out.push(tok.id);
      pos = m.index + m[1].length;
      if (tok.rstrip) {
        // consume leading whitespace of the next pre
        while (pos < text.length && /[ \t]/.test(text[pos])) pos++;
      }
      void post;
    }
    if (pos < text.length) out.push(text.slice(pos));
    return out;
  }

  /** A "word" for BPE is a metaspace-prefixed unit. Split on the replacement char. */
  _splitWords(metaspaced) {
    // Split on the replacement char but keep it as the leading char of each.
    const out = [];
    let cur = "";
    for (const ch of metaspaced) {
      if (ch === this.replacement) {
        if (cur) out.push(cur);
        cur = ch;
      } else {
        cur += ch;
      }
    }
    if (cur) out.push(cur);
    return out;
  }

  /** Standard BPE: greedy lowest-rank merge until no more merges apply. */
  _bpeEncode(word) {
    if (this.vocab.hasOwnProperty(word)) {
      return [this.vocab[word]];
    }
    // Start with characters
    let parts = Array.from(word);
    while (parts.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const key = parts[i] + " " + parts[i + 1];
        const rank = this.mergeRanks.get(key);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      parts = [
        ...parts.slice(0, bestIdx),
        parts[bestIdx] + parts[bestIdx + 1],
        ...parts.slice(bestIdx + 2),
      ];
    }
    // Map parts to ids; fall back to <unk> if missing.
    const unkId = this._unkId();
    return parts.map((p) => (this.vocab.hasOwnProperty(p) ? this.vocab[p] : unkId));
  }

  _unkId() {
    return (this.json.model && this.json.model.unk_token &&
            this.vocab[this.json.model.unk_token]) ?? 3;
  }

  _applyPostProcessor(ids) {
    if (!this.postProcessor) return ids;
    if (this.postProcessor.type !== "TemplateProcessing") return ids;
    // Single template = "$A </s>"  → append the </s> id.
    const out = ids.slice();
    for (const piece of this.postProcessor.single || []) {
      if (piece.SpecialToken) {
        const stId = piece.SpecialToken.id;
        const found = (this.postProcessor.special_tokens && this.postProcessor.special_tokens[stId]);
        if (found) out.push(found.ids[0]);
      }
    }
    return out;
  }

  /** Reverse the metaspace + drop specials. */
  decode(ids, { skip_special_tokens = true } = {}) {
    const tokens = [];
    for (const id of ids) {
      if (skip_special_tokens && this.specialIds.has(id)) continue;
      const tok = this.idToToken[id];
      if (tok !== undefined) tokens.push(tok);
    }
    let s = tokens.join("");
    s = s.split(this.replacement).join(" ");
    if (s.startsWith(" ")) s = s.slice(1);
    return s;
  }
}

export { IndicProcessor, IT2Tokenizer };
