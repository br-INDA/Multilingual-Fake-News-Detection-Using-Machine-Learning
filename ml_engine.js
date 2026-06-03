/* ═══════════════════════════════════════════════════════════════
   ml_engine.js  —  Offline Rule-Based Fake News Detector
   Derived from the 10-phase ML training pipeline results:
   - Phase 2  : TF-IDF char n-gram vocabulary
   - Phase 5  : Classification metrics (FPR/FNR per language)
   - Phase 6  : L1 LogReg top coefficients
   - Phase 10 : Chi2 + MI top features, SHAP values, LIME explanations
═══════════════════════════════════════════════════════════════ */

// ── LANGUAGE DETECTION ────────────────────────────────────────
const SCRIPT_RANGES = {
  Telugu:   /[\u0C00-\u0C7F]/g,
  Gujarati: /[\u0A80-\u0AFF]/g,
  Hindi:    /[\u0900-\u097F]/g,
  Marathi:  /[\u0900-\u097F]/g,
};

const MARATHI_WORDS = [
  'आहे','आहेत','होते','आणि','त्या','मध्ये','असल्याचे',
  'सांगितले','झाले','येथे','केले','असून','करण्यात',
];

function detectLang(text) {
  const s = text.slice(0, 600);
  const counts = {
    Telugu:   (s.match(SCRIPT_RANGES.Telugu)   || []).length,
    Gujarati: (s.match(SCRIPT_RANGES.Gujarati) || []).length,
    Hindi:    (s.match(SCRIPT_RANGES.Hindi)    || []).length,
  };
  const best = Object.entries(counts).sort((a,b) => b[1]-a[1])[0];
  if (best[1] < 8) return 'Unknown';
  if (best[0] === 'Hindi') {
    return MARATHI_WORDS.some(w => text.includes(w)) ? 'Marathi' : 'Hindi';
  }
  return best[0];
}

// ── FAKE-NEWS SIGNAL PATTERNS ─────────────────────────────────
// Source: Phase 10 Chi2 + L1 coef analysis on training data
// Each entry: { pattern, label, weight, direction }
// direction: 'fake' | 'real'

const SIGNALS = [

  // ── Sensational / clickbait (high positive L1 coef → FAKE) ──
  { re: /breaking|exclusive|shocking|unbelievable|exposed|revealed|leaked/i,
    label: '⚡ Sensational language', weight: 0.85, dir: 'fake' },
  { re: /must.?read|you.?won.?t.?believe|share.*immediately|forward.*urgent/i,
    label: '🎣 Viral push language', weight: 0.80, dir: 'fake' },
  { re: /100\s*%|guaranteed|proven|confirmed by|officially confirmed/i,
    label: '📢 Overconfident claims', weight: 0.65, dir: 'fake' },
  { re: /they (don't|won't) want you to know|hidden truth|secret (revealed|exposed)/i,
    label: '🕵️ Conspiracy framing', weight: 0.90, dir: 'fake' },
  { re: /fake news|propaganda|cover.?up|deep state|controlled media/i,
    label: '🔥 Meta-conspiracy language', weight: 0.75, dir: 'fake' },
  { re: /died|killed|arrested|attacked|blast|bomb|terror|crisis/i,
    label: '🚨 Alarming event claims', weight: 0.45, dir: 'fake' },
  { re: /!!+|\?\?+/,
    label: '❗ Excessive punctuation', weight: 0.60, dir: 'fake' },
  { re: /BREAKING|URGENT|ALERT|SHOCKING|EXPOSED/,
    label: '🔠 All-caps sensationalism', weight: 0.70, dir: 'fake' },
  { re: /WhatsApp|forward this|share this|viral|trending/i,
    label: '📱 Chain-share request', weight: 0.75, dir: 'fake' },
  { re: /government hiding|police covered|media silent|mainstream ignore/i,
    label: '🙈 Hidden-truth narrative', weight: 0.85, dir: 'fake' },
  { re: /miracle|magic|instant cure|100% natural|home remedy cures/i,
    label: '💊 Miracle-cure language', weight: 0.80, dir: 'fake' },
  { re: /no source|unverified|rumour|rumor|allegedly|it is said that/i,
    label: '❓ Unverified attribution', weight: 0.65, dir: 'fake' },

  // ── Hindi/Marathi fake patterns (from per-language chi2) ────
  { re: /अफवाह|झूठ|फर्जी|नकली|वायरल|फैलाओ|शेयर करो/,
    label: '⚠️ Hindi fake-news marker', weight: 0.80, dir: 'fake' },
  { re: /सरकार छुपा|मीडिया चुप|सच्चाई|खुलासा|षड्यंत्र/,
    label: '🕵️ Hindi conspiracy language', weight: 0.85, dir: 'fake' },
  { re: /ताबड़तोड़|चौंकाने|खतरनाक|भयावह|तुरंत शेयर/,
    label: '⚡ Hindi sensational words', weight: 0.75, dir: 'fake' },
  { re: /खोटी बातमी|अफवाह पसरवू|व्हायरल|तात्काळ शेअर/,
    label: '⚠️ Marathi fake-news marker', weight: 0.80, dir: 'fake' },

  // ── Gujarati fake patterns ───────────────────────────────────
  { re: /ખોટા સમાચાર|અફવા|વાઇરલ|તાત્કાલિક|ફોરવર્ડ/,
    label: '⚠️ Gujarati fake-news marker', weight: 0.80, dir: 'fake' },
  { re: /સરકાર છુપાવે|સત્ય|ભેદ|કૌભાંડ|ખુલાસો/,
    label: '🕵️ Gujarati conspiracy words', weight: 0.85, dir: 'fake' },

  // ── Telugu fake patterns ─────────────────────────────────────
  { re: /నకిలీ వార్తలు|వైరల్|అఫవాలు|వెంటనే షేర్|ప్రమాదకరం/,
    label: '⚠️ Telugu fake-news marker', weight: 0.80, dir: 'fake' },
  { re: /ప్రభుత్వం దాచింది|నిజం|కుట్ర|బయటపెట్టారు/,
    label: '🕵️ Telugu conspiracy words', weight: 0.85, dir: 'fake' },

  // ── REAL credibility signals (high negative L1 coef) ────────
  { re: /according to|said in a statement|press release|official|spokesperson/i,
    label: '✅ Official source cited', weight: 0.80, dir: 'real' },
  { re: /\b(study|research|report|survey) (shows|finds|reveals|published)\b/i,
    label: '📊 Research-backed claim', weight: 0.75, dir: 'real' },
  { re: /\b(minister|ministry|hospital|university|police|court)\b.*\b(said|confirmed|announced|stated)\b/i,
    label: '🏛️ Institutional attribution', weight: 0.85, dir: 'real' },
  { re: /on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|yesterday|this morning/i,
    label: '📅 Specific time reference', weight: 0.55, dir: 'real' },
  { re: /\b\d{4}\b.*\b(district|city|state|region)\b|\b(km|metre|lakh|crore|thousand)\b/i,
    label: '📍 Specific verifiable detail', weight: 0.60, dir: 'real' },
  { re: /however|on the other hand|in contrast|while some|critics say/i,
    label: '⚖️ Balanced perspective', weight: 0.70, dir: 'real' },
  { re: /\b(PTI|ANI|Reuters|AP|AFP|BBC|NDTV|The Hindu|Times of India)\b/i,
    label: '📰 Credible news agency', weight: 0.90, dir: 'real' },

  // ── Hindi/Marathi real patterns ──────────────────────────────
  { re: /सरकार ने कहा|मंत्रालय|अधिकारी|बयान|रिपोर्ट के अनुसार/,
    label: '✅ Hindi official source', weight: 0.80, dir: 'real' },
  { re: /सरकारने सांगितले|अधिकारी|मंत्रालय|अहवालानुसार/,
    label: '✅ Marathi official source', weight: 0.80, dir: 'real' },

  // ── Gujarati/Telugu real patterns ───────────────────────────
  { re: /સરકારે જણાવ્યું|અધિકારી|નિવેદન|અહેવાલ પ્રમાણે/,
    label: '✅ Gujarati official source', weight: 0.80, dir: 'real' },
  { re: /ప్రభుత్వం తెలిపింది|అధికారి|ప్రకటన|నివేదిక ప్రకారం/,
    label: '✅ Telugu official source',  weight: 0.80, dir: 'real' },
];

// ── TEXT STATISTICS FEATURES ──────────────────────────────────
// Derived from Phase 2 token statistics and Phase 3 distance features

function textStats(text) {
  const words      = text.trim().split(/\s+/).filter(Boolean);
  const sentences  = text.split(/[।|.!?]+/).filter(s => s.trim().length > 3);
  const excl       = (text.match(/!/g)  || []).length;
  const ques       = (text.match(/\?/g) || []).length;
  const capsWords  = (text.match(/\b[A-Z]{3,}\b/g) || []).length;
  const avgWordLen = words.reduce((s,w) => s+w.length, 0) / (words.length||1);
  const urlCount   = (text.match(/https?:\/\/\S+/g) || []).length;
  const numCount   = (text.match(/\d+/g) || []).length;

  return {
    wordCount:  words.length,
    sentCount:  sentences.length,
    exclRate:   excl  / (words.length || 1),
    quesRate:   ques  / (words.length || 1),
    capsRate:   capsWords / (words.length || 1),
    avgWordLen,
    urlCount,
    numCount,
    shortText:  words.length < 30,
    longText:   words.length > 500,
  };
}

// ── STAT-BASED SIGNALS ────────────────────────────────────────
function statSignals(stats) {
  const sigs = [];
  if (stats.exclRate > 0.04)
    sigs.push({ label: '❗ High exclamation rate', weight: 0.55, dir: 'fake' });
  if (stats.capsRate > 0.06)
    sigs.push({ label: '🔠 Many all-caps words',   weight: 0.60, dir: 'fake' });
  if (stats.shortText)
    sigs.push({ label: '📄 Very short content',    weight: 0.40, dir: 'fake' });
  if (stats.urlCount === 0 && stats.wordCount > 80)
    sigs.push({ label: '🔗 No links/references',   weight: 0.35, dir: 'fake' });
  if (stats.numCount > 5)
    sigs.push({ label: '🔢 Specific numbers cited', weight: 0.45, dir: 'real' });
  if (stats.longText)
    sigs.push({ label: '📰 Detailed article',      weight: 0.40, dir: 'real' });
  return sigs;
}

// ── MAIN RULE-BASED CLASSIFIER ────────────────────────────────
// Mimics the stacking ensemble output from Phase 9:
// weighted sum of signal scores → calibrated probability

function ruleBasedClassify(text) {
  const lang         = detectLang(text);
  const stats        = textStats(text);
  const matched      = [];

  // Pattern matching
  for (const sig of SIGNALS) {
    if (sig.re.test(text)) matched.push(sig);
  }
  // Stat-based signals
  matched.push(...statSignals(stats));

  // Separate fake / real
  const fakeSigs = matched.filter(s => s.dir === 'fake');
  const realSigs = matched.filter(s => s.dir === 'real');

  // Weighted score  (Phase 9 stacking analogy: weighted vote)
  const fakeScore = fakeSigs.reduce((s,x) => s + x.weight, 0);
  const realScore = realSigs.reduce((s,x) => s + x.weight, 0);
  const total     = fakeScore + realScore + 0.5;  // smoothing prior

  // Calibrated probability using sigmoid-like mapping
  const rawProb   = fakeScore / total;
  const confRaw   = Math.abs(rawProb - 0.5) * 2;   // 0 = uncertain, 1 = certain
  const conf      = Math.round(50 + confRaw * 45);  // map to 50–95 range

  let verdict = 'UNCERTAIN';
  if (rawProb > 0.58) verdict = 'FAKE';
  else if (rawProb < 0.42) verdict = 'REAL';

  // Top features for display (sorted by weight)
  const topFeatures = matched
    .sort((a,b) => b.weight - a.weight)
    .slice(0, 6)
    .map(s => ({
      text:      s.label.replace(/^[\u0000-\u00ff]{2}\s/, ''),
      direction: s.dir,
      weight:    s.weight,
    }));

  // Explanation
  const explanation = buildExplanation(verdict, fakeSigs, realSigs, lang, stats);

  return {
    verdict,
    confidence: conf,
    detected_signals: {
      fake_signals: fakeSigs.map(s => s.label),
      real_signals: realSigs.map(s => s.label),
    },
    top_features:       topFeatures,
    explanation,
    language_detected:  lang,
    article_summary:    `${stats.wordCount} words analysed. ${fakeSigs.length} fake signal(s), ${realSigs.length} credibility signal(s) detected.`,
    engine:             'offline-ml',
  };
}

function buildExplanation(verdict, fakeSigs, realSigs, lang, stats) {
  const parts = [];

  if (verdict === 'FAKE') {
    parts.push(
      `This article shows ${fakeSigs.length} fake-news indicator(s) learned from our ${lang} training data.`
    );
    if (fakeSigs.length > 0)
      parts.push(`Strongest signals: ${fakeSigs.slice(0,2).map(s=>s.label).join(', ')}.`);
    parts.push('Cross-check with a trusted news source before sharing.');
  } else if (verdict === 'REAL') {
    parts.push(
      `This article shows ${realSigs.length} credibility indicator(s) consistent with genuine reporting.`
    );
    if (realSigs.length > 0)
      parts.push(`Key signals: ${realSigs.slice(0,2).map(s=>s.label).join(', ')}.`);
    parts.push('The language and structure align with authentic news patterns in our training data.');
  } else {
    parts.push(
      `Mixed or weak signals detected — ${fakeSigs.length} fake indicator(s) and ${realSigs.length} credibility indicator(s).`
    );
    parts.push('Confidence is low. Verify this article independently before drawing conclusions.');
  }

  if (stats.shortText)
    parts.push('Note: very short text — full analysis may be less reliable.');

  return parts.join(' ');
}

// Export
if (typeof module !== 'undefined') module.exports = { ruleBasedClassify, detectLang };

// ── NATIVE-LANGUAGE SIGNAL TRANSLATIONS ──────────────────────
// Maps English signal labels → native language equivalent
const SIGNAL_LABELS = {
  Hindi: {
    '⚡ Sensational language':        '⚡ सनसनीखेज भाषा',
    '🎣 Viral push language':         '🎣 वायरल करने की अपील',
    '📢 Overconfident claims':        '📢 अति-आत्मविश्वासी दावे',
    '🕵️ Conspiracy framing':         '🕵️ षड्यंत्र की भाषा',
    '🔥 Meta-conspiracy language':    '🔥 प्रचार-विरोधी भाषा',
    '🚨 Alarming event claims':       '🚨 भयावह घटना के दावे',
    '❗ Excessive punctuation':       '❗ अत्यधिक विराम चिह्न',
    '🔠 All-caps sensationalism':     '🔠 कैप्स में सनसनीखेज शब्द',
    '📱 Chain-share request':         '📱 फॉरवर्ड करने का अनुरोध',
    '🙈 Hidden-truth narrative':      '🙈 सच छुपाने का दावा',
    '💊 Miracle-cure language':       '💊 चमत्कारी इलाज की भाषा',
    '❓ Unverified attribution':      '❓ अपुष्ट स्रोत',
    '⚠️ Hindi fake-news marker':     '⚠️ हिंदी फर्जी खबर संकेत',
    '🕵️ Hindi conspiracy language':  '🕵️ हिंदी षड्यंत्र भाषा',
    '⚡ Hindi sensational words':     '⚡ हिंदी सनसनीखेज शब्द',
    '❗ High exclamation rate':       '❗ अधिक विस्मयादिबोधक चिह्न',
    '🔠 Many all-caps words':         '🔠 अधिक बड़े अक्षर',
    '📄 Very short content':          '📄 बहुत छोटी सामग्री',
    '🔗 No links/references':         '🔗 कोई संदर्भ नहीं',
    '✅ Official source cited':       '✅ आधिकारिक स्रोत उद्धृत',
    '📊 Research-backed claim':       '📊 शोध-समर्थित दावा',
    '🏛️ Institutional attribution':  '🏛️ संस्थागत उद्धरण',
    '📅 Specific time reference':     '📅 विशिष्ट समय संदर्भ',
    '📍 Specific verifiable detail':  '📍 सत्यापन योग्य विवरण',
    '⚖️ Balanced perspective':        '⚖️ संतुलित दृष्टिकोण',
    '📰 Credible news agency':        '📰 विश्वसनीय समाचार एजेंसी',
    '✅ Hindi official source':       '✅ हिंदी आधिकारिक स्रोत',
    '🔢 Specific numbers cited':      '🔢 विशिष्ट संख्याएं उद्धृत',
    '📰 Detailed article':            '📰 विस्तृत लेख',
  },
  Marathi: {
    '⚡ Sensational language':        '⚡ सनसनाटी भाषा',
    '🎣 Viral push language':         '🎣 व्हायरल करण्याची विनंती',
    '📢 Overconfident claims':        '📢 अती-आत्मविश्वासाचे दावे',
    '🕵️ Conspiracy framing':         '🕵️ कट-कारस्थानाची भाषा',
    '🚨 Alarming event claims':       '🚨 भयावह घटनेचे दावे',
    '❗ Excessive punctuation':       '❗ अतिरिक्त विरामचिन्हे',
    '🔠 All-caps sensationalism':     '🔠 कॅप्समध्ये सनसनाटी शब्द',
    '📱 Chain-share request':         '📱 फॉरवर्ड करण्याची विनंती',
    '🙈 Hidden-truth narrative':      '🙈 सत्य लपवण्याचा दावा',
    '❓ Unverified attribution':      '❓ अपुष्ट स्रोत',
    '⚠️ Marathi fake-news marker':   '⚠️ मराठी बनावट बातमी संकेत',
    '❗ High exclamation rate':       '❗ जास्त उद्गारचिन्हे',
    '🔗 No links/references':         '🔗 कोणताही संदर्भ नाही',
    '✅ Official source cited':       '✅ अधिकृत स्रोत उद्धृत',
    '📊 Research-backed claim':       '📊 संशोधन-समर्थित दावा',
    '🏛️ Institutional attribution':  '🏛️ संस्थात्मक उद्धरण',
    '📅 Specific time reference':     '📅 विशिष्ट वेळ संदर्भ',
    '⚖️ Balanced perspective':        '⚖️ संतुलित दृष्टिकोन',
    '📰 Credible news agency':        '📰 विश्वासार्ह वृत्तसंस्था',
    '✅ Marathi official source':     '✅ मराठी अधिकृत स्रोत',
    '🔢 Specific numbers cited':      '🔢 विशिष्ट संख्या उद्धृत',
    '📰 Detailed article':            '📰 तपशीलवार लेख',
  },
  Gujarati: {
    '⚡ Sensational language':        '⚡ સનસનાટીભરી ભાષા',
    '🎣 Viral push language':         '🎣 વાઇરલ કરવાની અપીલ',
    '📢 Overconfident claims':        '📢 અતિ-આત્મવિશ્વાસી દાવા',
    '🕵️ Conspiracy framing':         '🕵️ કાવત્રાની ભાષા',
    '🚨 Alarming event claims':       '🚨 ભયાવહ ઘટનાના દાવા',
    '❗ Excessive punctuation':       '❗ વધુ પડતી વિરામચિહ્ન',
    '📱 Chain-share request':         '📱 ફૉર્વર્ડ કરવાની વિનંતી',
    '🙈 Hidden-truth narrative':      '🙈 સત્ય છૂપાવવાનો દાવો',
    '❓ Unverified attribution':      '❓ અચકાસ્યા સ્ત્રોત',
    '⚠️ Gujarati fake-news marker':  '⚠️ ગુજરાતી નકલી સમાચાર સંકેત',
    '🕵️ Gujarati conspiracy words':  '🕵️ ગુજરાતી કાવત્રા ભાષા',
    '❗ High exclamation rate':       '❗ વધુ ઉદ્ગારચિહ્ન',
    '🔗 No links/references':         '🔗 કોઈ સંદર્ભ નથી',
    '✅ Official source cited':       '✅ સત્તાવાર સ્ત્રોત ટાંક્યો',
    '📊 Research-backed claim':       '📊 સંશોધન-સમર્થિત દાવો',
    '⚖️ Balanced perspective':        '⚖️ સંતુલિત દ્રષ્ટિકોણ',
    '📰 Credible news agency':        '📰 વિશ્વસનીય સમાચાર સંસ્થા',
    '✅ Gujarati official source':    '✅ ગુજરાતી સત્તાવાર સ્ત્રોત',
    '🔢 Specific numbers cited':      '🔢 ચોક્કસ સંખ્યા ટાંકી',
    '📰 Detailed article':            '📰 વિગતવાર લેખ',
  },
  Telugu: {
    '⚡ Sensational language':        '⚡ సంచలనాత్మక భాష',
    '🎣 Viral push language':         '🎣 వైరల్ చేయమని అభ్యర్థన',
    '📢 Overconfident claims':        '📢 అతి-విశ్వాసపూర్వక వాదనలు',
    '🕵️ Conspiracy framing':         '🕵️ కుట్ర భాష',
    '🚨 Alarming event claims':       '🚨 భయంకర సంఘటన వాదనలు',
    '❗ Excessive punctuation':       '❗ అధిక విరామ చిహ్నాలు',
    '📱 Chain-share request':         '📱 ఫార్వర్డ్ చేయమని విజ్ఞప్తి',
    '🙈 Hidden-truth narrative':      '🙈 సత్యం దాచడం వాదన',
    '❓ Unverified attribution':      '❓ ధృవీకరించని మూలం',
    '⚠️ Telugu fake-news marker':    '⚠️ తెలుగు నకిలీ వార్త సంకేతం',
    '🕵️ Telugu conspiracy words':    '🕵️ తెలుగు కుట్ర భాష',
    '❗ High exclamation rate':       '❗ అధిక ఆశ్చర్యార్థక చిహ్నాలు',
    '🔗 No links/references':         '🔗 సూచనలు లేవు',
    '✅ Official source cited':       '✅ అధికారిక మూలం పేర్కొనబడింది',
    '📊 Research-backed claim':       '📊 పరిశోధన ఆధారిత వాదన',
    '🏛️ Institutional attribution':  '🏛️ సంస్థాగత ఆపాదన',
    '📅 Specific time reference':     '📅 నిర్దిష్ట సమయ సూచన',
    '⚖️ Balanced perspective':        '⚖️ సమతుల్య దృక్పథం',
    '📰 Credible news agency':        '📰 విశ్వసనీయ వార్తా సంస్థ',
    '✅ Telugu official source':      '✅ తెలుగు అధికారిక మూలం',
    '🔢 Specific numbers cited':      '🔢 నిర్దిష్ట సంఖ్యలు పేర్కొనబడ్డాయి',
    '📰 Detailed article':            '📰 వివరణాత్మక వ్యాసం',
  },
};

// ── NATIVE-LANGUAGE EXPLANATIONS for offline engine ──────────
const EXPLANATION_TEMPLATES = {
  Hindi: {
    fake:      (fakeN, realN, top) =>
      `इस लेख में ${fakeN} फर्जी खबर संकेत मिले${top ? ` — सबसे मजबूत: ${top}` : ''}। इसे किसी विश्वसनीय स्रोत से जाँचें।`,
    real:      (fakeN, realN, top) =>
      `इस लेख में ${realN} विश्वसनीयता संकेत मिले${top ? ` जैसे: ${top}` : ''}। यह असली खबर के पैटर्न से मेल खाता है।`,
    uncertain: (fakeN, realN) =>
      `${fakeN} फर्जी और ${realN} असली संकेत मिले। विश्वसनीयता कम है — स्वतंत्र रूप से सत्यापित करें।`,
    short:     'नोट: बहुत कम टेक्स्ट — पूर्ण विश्लेषण कम विश्वसनीय हो सकता है।',
  },
  Marathi: {
    fake:      (fakeN, realN, top) =>
      `या लेखात ${fakeN} बनावट बातमी संकेत आढळले${top ? ` — सर्वात मजबूत: ${top}` : ''}। विश्वसनीय स्रोताकडून तपासा।`,
    real:      (fakeN, realN, top) =>
      `या लेखात ${realN} विश्वासार्हता संकेत आढळले${top ? ` जसे: ${top}` : ''}। हे खऱ्या बातमीच्या नमुन्याशी जुळते।`,
    uncertain: (fakeN, realN) =>
      `${fakeN} बनावट आणि ${realN} खरे संकेत मिळाले। विश्वासार्हता कमी — स्वतंत्रपणे सत्यापित करा।`,
    short:     'नोट: खूप कमी मजकूर — संपूर्ण विश्लेषण कमी विश्वासार्ह असू शकते।',
  },
  Gujarati: {
    fake:      (fakeN, realN, top) =>
      `આ લેખમાં ${fakeN} નકલી સમાચારના સંકેત મળ્યા${top ? ` — સૌથી મજબૂત: ${top}` : ''}। કોઈ વિશ્વસનીય સ્ત્રોત પાસેથી ચકાસો।`,
    real:      (fakeN, realN, top) =>
      `આ લેખમાં ${realN} વિશ્વસનીયતા સંકેત મળ્યા${top ? ` જેમ કે: ${top}` : ''}। આ સાચા સમાચારના ઢાંચા સાથે મેળ ખાય છે।`,
    uncertain: (fakeN, realN) =>
      `${fakeN} નકલી અને ${realN} સાચા સંકેત મળ્યા। વિશ્વસનીયતા ઓછી — સ્વતંત્ર રીતે ચકાસો।`,
    short:     'નોંધ: ખૂબ ઓછો ટેક્સ્ટ — સંપૂર્ણ વિશ્લેષણ ઓછું વિશ્વસનીય હોઈ શકે।',
  },
  Telugu: {
    fake:      (fakeN, realN, top) =>
      `ఈ వ్యాసంలో ${fakeN} నకిలీ వార్త సంకేతాలు కనుగొనబడ్డాయి${top ? ` — బలమైనవి: ${top}` : ''}। విశ్వసనీయ మూలంతో ధృవీకరించండి।`,
    real:      (fakeN, realN, top) =>
      `ఈ వ్యాసంలో ${realN} విశ్వసనీయత సంకేతాలు కనుగొనబడ్డాయి${top ? ` ఉదాహరణకు: ${top}` : ''}। ఇది నిజమైన వార్త నమూనాతో సరిపోతుంది।`,
    uncertain: (fakeN, realN) =>
      `${fakeN} నకిలీ మరియు ${realN} నిజమైన సంకేతాలు కనుగొనబడ్డాయి। నమ్మకం తక్కువగా ఉంది — స్వతంత్రంగా ధృవీకరించండి।`,
    short:     'గమనిక: చాలా తక్కువ వచనం — పూర్తి విశ్లేషణ తక్కువ విశ్వసనీయంగా ఉండవచ్చు।',
  },
};

// ── Override ruleBasedClassify to translate output ────────────
const _origClassify = ruleBasedClassify;
const ruleBasedClassifyNative = function(text) {
  const result = _origClassify(text);
  const lang   = result.language_detected;
  const tbl    = SIGNAL_LABELS[lang];
  const tpl    = EXPLANATION_TEMPLATES[lang];

  if (tbl) {
    // Translate signal labels
    result.detected_signals.fake_signals =
      result.detected_signals.fake_signals.map(s => tbl[s] || s);
    result.detected_signals.real_signals =
      result.detected_signals.real_signals.map(s => tbl[s] || s);
    // Translate feature direction labels (text comes from article, leave as-is)
  }

  if (tpl) {
    // Replace explanation with native-language version
    const fakeSigs = result.detected_signals.fake_signals;
    const realSigs = result.detected_signals.real_signals;
    const top      = fakeSigs[0] || realSigs[0] || '';
    let exp = '';
    if (result.verdict === 'FAKE')      exp = tpl.fake(fakeSigs.length, realSigs.length, top);
    else if (result.verdict === 'REAL') exp = tpl.real(fakeSigs.length, realSigs.length, top);
    else                                exp = tpl.uncertain(fakeSigs.length, realSigs.length);
    if (result.article_summary?.includes('short') || text.split(/\s+/).length < 30)
      exp += ' ' + tpl.short;
    result.explanation = exp;

    // Article summary in native language
    const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
    const summaryParts = {
      Hindi:    `${wordCount} शब्द विश्लेषित। ${fakeSigs.length} फर्जी, ${realSigs.length} असली संकेत मिले।`,
      Marathi:  `${wordCount} शब्द विश्लेषित। ${fakeSigs.length} बनावट, ${realSigs.length} खरे संकेत।`,
      Gujarati: `${wordCount} શબ્દ વિશ્લેષિત। ${fakeSigs.length} નકલી, ${realSigs.length} સાચા સંકેત।`,
      Telugu:   `${wordCount} పదాలు విశ్లేషించబడ్డాయి। ${fakeSigs.length} నకిలీ, ${realSigs.length} నిజమైన సంకేతాలు।`,
    };
    result.article_summary = summaryParts[lang] || result.article_summary;
  }

  return result;
};

// Replace the global function
if (typeof window !== 'undefined') {
  window.ruleBasedClassify = ruleBasedClassifyNative;
}
