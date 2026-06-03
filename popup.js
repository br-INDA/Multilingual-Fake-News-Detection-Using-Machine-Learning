/* ═══════════════════════════════════════════════════════════════
   popup.js  —  FakeShield v2  (multilingual UI output)
   All result text — verdicts, signals, explanations, labels —
   are displayed in the detected article language.
═══════════════════════════════════════════════════════════════ */

// ── UI translations per language ──────────────────────────────
const UI = {
  Hindi: {
    verdict_fake:      'संभावित फ़र्ज़ी खबर',
    verdict_real:      'संभावित असली खबर',
    verdict_uncertain: 'अनिश्चित — जाँच करें',
    confidence:        'विश्वसनीयता',
    signals_title:     'ML संकेत मिले',
    features_title:    'शीर्ष भविष्यसूचक विशेषताएं',
    explanation_title: 'स्पष्टीकरण',
    to_fake:           '→ फर्जी',
    to_real:           '→ असली',
    copy_btn:          '📋 रिपोर्ट कॉपी करें',
    reset_btn:         '🔄 रीसेट',
    no_signals:        'कोई विशिष्ट संकेत नहीं मिला',
    no_features:       'फ़ीचर डेटा उपलब्ध नहीं',
    engine_tag_fallback: 'ऑफलाइन ML (फ़ॉलबैक)',
    bias_prefix:       '📊 मॉडल नोट (हिंदी):',
  },
  Marathi: {
    verdict_fake:      'संभाव्य बनावट बातमी',
    verdict_real:      'संभाव्य खरी बातमी',
    verdict_uncertain: 'अनिश्चित — तपासा',
    confidence:        'विश्वासार्हता',
    signals_title:     'ML संकेत सापडले',
    features_title:    'शीर्ष भविष्यसूचक वैशिष्ट्ये',
    explanation_title: 'स्पष्टीकरण',
    to_fake:           '→ बनावट',
    to_real:           '→ खरे',
    copy_btn:          '📋 अहवाल कॉपी करा',
    reset_btn:         '🔄 रीसेट',
    no_signals:        'कोणतेही विशिष्ट संकेत आढळले नाहीत',
    no_features:       'फीचर डेटा उपलब्ध नाही',
    engine_tag_fallback: 'ऑफलाइन ML (फॉलबॅक)',
    bias_prefix:       '📊 मॉडेल नोट (मराठी):',
  },
  Gujarati: {
    verdict_fake:      'સંભવિત નકલી સમાચાર',
    verdict_real:      'સંભવિત સાચા સમાચાર',
    verdict_uncertain: 'અનિશ્ચિત — ચકાસો',
    confidence:        'વિશ્વાસ',
    signals_title:     'ML સંકેતો મળ્યા',
    features_title:    'ટોચની ભવિષ્યસૂચક વિશેષતાઓ',
    explanation_title: 'સ્પષ્ટીકરણ',
    to_fake:           '→ નકલી',
    to_real:           '→ સાચા',
    copy_btn:          '📋 રિપોર્ટ કૉપિ કરો',
    reset_btn:         '🔄 રીસેટ',
    no_signals:        'કોઈ ચોક્કસ સંકેત મળ્યો નથી',
    no_features:       'ફીચર ડેટા ઉપલબ્ધ નથી',
    engine_tag_fallback: 'ઑફલાઇન ML (ફૉલબૅક)',
    bias_prefix:       '📊 મૉડલ નોટ (ગુજરાતી):',
  },
  Telugu: {
    verdict_fake:      'నకిలీ వార్త అని అనుమానం',
    verdict_real:      'నిజమైన వార్త అని అనుమానం',
    verdict_uncertain: 'అనిశ్చితం — ధృవీకరించండి',
    confidence:        'నమ్మకం',
    signals_title:     'ML సంకేతాలు కనుగొనబడ్డాయి',
    features_title:    'అగ్రశ్రేణి అంచనా లక్షణాలు',
    explanation_title: 'వివరణ',
    to_fake:           '→ నకిలీ',
    to_real:           '→ నిజమైన',
    copy_btn:          '📋 నివేదిక కాపీ చేయండి',
    reset_btn:         '🔄 రీసెట్',
    no_signals:        'నిర్దిష్ట సంకేతాలు కనుగొనబడలేదు',
    no_features:       'ఫీచర్ డేటా అందుబాటులో లేదు',
    engine_tag_fallback: 'ఆఫ్‌లైన్ ML (ఫాల్‌బ్యాక్)',
    bias_prefix:       '📊 మోడల్ గమనిక (తెలుగు):',
  },
  Unknown: {
    verdict_fake:      'Likely Fake News',
    verdict_real:      'Likely Genuine',
    verdict_uncertain: 'Uncertain — Verify',
    confidence:        'Confidence',
    signals_title:     'ML SIGNALS DETECTED',
    features_title:    'TOP PREDICTIVE FEATURES',
    explanation_title: 'EXPLANATION',
    to_fake:           '→Fake',
    to_real:           '→Real',
    copy_btn:          '📋 Copy Report',
    reset_btn:         '🔄 Reset',
    no_signals:        'No specific signals detected',
    no_features:       'No feature data',
    engine_tag_fallback: 'Offline ML (fallback)',
    bias_prefix:       '📊 Model note:',
  },
};

// ── Per-language bias notes (in native script) ────────────────
const LANG_NOTES = {
  Hindi:    'मॉडल नोट (Phase 10): FNR=0.063 — कुछ फर्जी खबरें छूट सकती हैं। महत्वपूर्ण समाचार को किसी विश्वसनीय स्रोत से जाँचें।',
  Marathi:  'मराठीसाठी आमच्या प्रशिक्षणात सर्वोत्तम F1 मिळाला. उच्च-विश्वासार्हता परिणाम विश्वसनीय आहेत।',
  Gujarati: 'ગુજરાતીના ઓછા ટ્રેનિંગ ડેટાને કારણે અનિશ્ચિત પરિણામો ધ્યાનથી વાંચો.',
  Telugu:   'తెలుగు మోడల్‌లో FNR అత్యధికంగా ఉంది. తెలుగు ఫలితాలను స్వతంత్రంగా ధృవీకరించడం మంచిది.',
};

// ── Engine metadata ───────────────────────────────────────────
const ENGINE_META = {
  offline:   { label: 'Offline ML',      badge: 'badge-offline',   keyHint: '' },
  groq:      { label: 'Groq (Llama-3)',  badge: 'badge-groq',      keyHint: 'Groq API key starts with gsk_…' },
  gemini:    { label: 'Gemini Flash',    badge: 'badge-gemini',    keyHint: 'Google AI Studio key starts with AIza…' },
  anthropic: { label: 'Claude (Anthropic)', badge: 'badge-anthropic', keyHint: 'Anthropic key starts with sk-ant-…' },
};

// ── State ─────────────────────────────────────────────────────
let activeTab    = 'page';
let activeLang   = 'auto';
let activeEngine = 'offline';
let isAnalysing  = false;
let lastReport   = null;
let currentLang  = 'Unknown';  // detected language for current result

// ── DOM refs ──────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  bindEvents();
  updateEngineBadge();
});

function bindEvents() {
  document.querySelectorAll('.pill').forEach(p =>
    p.addEventListener('click', () => {
      document.querySelectorAll('.pill').forEach(x => x.classList.remove('active'));
      p.classList.add('active');
      activeLang = p.dataset.lang;
    })
  );
  $('tabPage').addEventListener('click',     () => switchTab('page'));
  $('tabSelected').addEventListener('click', () => switchTab('selected'));
  $('tabPaste').addEventListener('click',    () => switchTab('paste'));
  $('analyseBtn').addEventListener('click',  runAnalysis);
  $('settingsBtn').addEventListener('click',   () => $('settingsPanel').classList.remove('hidden'));
  $('closeSettings').addEventListener('click', () => $('settingsPanel').classList.add('hidden'));
  $('saveSettings').addEventListener('click',  saveSettings);
  document.querySelectorAll('input[name="engine"]').forEach(r =>
    r.addEventListener('change', () => { activeEngine = r.value; updateEngineUI(); })
  );
  document.querySelectorAll('.engine-card').forEach(card =>
    card.addEventListener('click', () => {
      const radio = card.querySelector('input[type="radio"]');
      if (radio) { radio.checked = true; radio.dispatchEvent(new Event('change')); }
    })
  );
  $('copyBtn').addEventListener('click',  copyReport);
  $('resetBtn').addEventListener('click', resetUI);
}

// ── Settings ──────────────────────────────────────────────────
async function loadSettings() {
  const s = await chrome.storage.local.get(['engine','apiKey','sensitivity']);
  if (s.engine) {
    activeEngine = s.engine;
    const radio  = document.querySelector(`input[value="${s.engine}"]`);
    if (radio) radio.checked = true;
  }
  if (s.apiKey)      $('apiKeyInput').value       = s.apiKey;
  if (s.sensitivity) $('sensitivitySelect').value = s.sensitivity;
  updateEngineUI();
}

async function saveSettings() {
  await chrome.storage.local.set({
    engine:      activeEngine,
    apiKey:      $('apiKeyInput').value.trim(),
    sensitivity: $('sensitivitySelect').value,
  });
  updateEngineBadge();
  $('saveMsg').classList.remove('hidden');
  setTimeout(() => $('saveMsg').classList.add('hidden'), 1800);
}

function updateEngineUI() {
  const isOffline = (activeEngine === 'offline');
  $('apiKeySection').classList.toggle('hidden', isOffline);
  if (!isOffline) {
    const meta = ENGINE_META[activeEngine];
    $('apiKeyLabel').textContent = meta.label + ' API Key';
    $('apiKeyHint').textContent  = meta.keyHint;
  }
  document.querySelectorAll('.engine-card').forEach(c =>
    c.classList.toggle('selected', c.dataset.engine === activeEngine)
  );
}

function updateEngineBadge() {
  const meta  = ENGINE_META[activeEngine] || ENGINE_META.offline;
  const badge = $('engineBadge');
  badge.textContent = meta.label;
  badge.className   = `badge ${meta.badge}`;
  $('engineStatus').textContent = activeEngine === 'offline'
    ? '✓ Ready (no internet needed)' : '';
}

// ── Source tab ────────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  ['page','selected','paste'].forEach(t => {
    const btn = $(`tab${t.charAt(0).toUpperCase()+t.slice(1)}`);
    if (btn) btn.classList.toggle('active', t === tab);
  });
  $('pasteArea').classList.toggle('hidden', tab !== 'paste');
  $('btnLabel').textContent = tab === 'page' ? 'Analyse Article' : 'Analyse Text';
}

// ── Text extraction ───────────────────────────────────────────
async function getText() {
  if (activeTab === 'paste') {
    const t = $('pasteArea').value.trim();
    if (!t) throw new Error('Please paste some text to analyse.');
    return t;
  }
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (activeTab === 'selected') {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection()?.toString().trim() || '',
    });
    if (!result) throw new Error('No text selected. Highlight text on the page first.');
    return result;
  }
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const sel = ['article','[role="main"]','.article-body','.post-content',
                   '.story-body','.entry-content','main'];
      let node = null;
      for (const s of sel) {
        const el = document.querySelector(s);
        if (el && el.innerText.trim().length > 100) { node = el; break; }
      }
      if (!node) node = document.body;
      const paras = Array.from(node.querySelectorAll('p,h1,h2,h3'))
        .map(e => e.innerText.trim()).filter(t => t.length > 20);
      return (document.title + '. ' + paras.join(' ')).slice(0, 4000);
    },
  });
  if (!result || result.length < 50)
    throw new Error('Not enough text found on this page. Try Paste Text mode instead.');
  return result;
}

// ── Main analysis ─────────────────────────────────────────────
async function runAnalysis() {
  if (isAnalysing) return;
  let text;
  try { text = await getText(); }
  catch (e) { showError(e.message); return; }

  const lang = activeLang === 'auto'
    ? (typeof detectLang === 'function' ? detectLang(text) : 'Unknown')
    : activeLang;

  currentLang = lang;
  applyUILanguage(lang);   // ← update UI labels immediately

  setLoading(true);
  $('resultCard').classList.add('hidden');
  $('errorBox').classList.add('hidden');

  try {
    let result;
    if (activeEngine === 'offline') {
      result = ruleBasedClassify(text);
    } else {
      const { apiKey, sensitivity } =
        await chrome.storage.local.get(['apiKey','sensitivity']);
      if (!apiKey) throw new Error(
        `No API key for ${ENGINE_META[activeEngine].label}.\nOpen Settings ⚙️ or switch to Offline ML.`
      );
      if (activeEngine === 'groq')      result = await callGroq(text, lang, apiKey, sensitivity);
      else if (activeEngine === 'gemini')    result = await callGemini(text, lang, apiKey, sensitivity);
      else if (activeEngine === 'anthropic') result = await callAnthropic(text, lang, apiKey, sensitivity);
    }

    lastReport = { result, lang, snippet: text.slice(0, 200) };
    renderResult(result, lang);

  } catch (err) {
    if (activeEngine !== 'offline') {
      try {
        const fallback = ruleBasedClassify(text);
        fallback._fallback = true;
        lastReport = { result: fallback, lang, snippet: text.slice(0,200) };
        renderResult(fallback, lang);
        showError(`⚠️ ${ENGINE_META[activeEngine].label} API failed: ${err.message}\nShowing offline ML result.`);
        return;
      } catch (_) {}
    }
    showError(err.message || 'An unexpected error occurred.');
  } finally {
    setLoading(false);
  }
}

// ── Apply native-language labels to UI elements ───────────────
function applyUILanguage(lang) {
  const t = UI[lang] || UI.Unknown;
  // Section headers
  const sigTitle  = document.querySelector('.section-label[data-key="signals"]');
  const featTitle = document.querySelector('.section-label[data-key="features"]');
  const expTitle  = document.querySelector('.section-label[data-key="explanation"]');
  const confLabel = document.querySelector('.conf-label');
  if (sigTitle)  sigTitle.textContent  = t.signals_title;
  if (featTitle) featTitle.textContent = t.features_title;
  if (expTitle)  expTitle.textContent  = t.explanation_title;
  if (confLabel) confLabel.textContent = t.confidence;
  // Buttons
  const copyBtn  = $('copyBtn');
  const resetBtn = $('resetBtn');
  if (copyBtn)  copyBtn.textContent  = t.copy_btn;
  if (resetBtn) resetBtn.textContent = t.reset_btn;
}

// ── API calls ─────────────────────────────────────────────────
async function callGroq(text, lang, apiKey, sensitivity) {
  const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile', max_tokens: 900, temperature: 0.1,
      messages: [
        { role: 'system', content: buildPrompt(lang, sensitivity) },
        { role: 'user',   content: `Analyse this ${lang} news article:\n\n${text.slice(0,3000)}` },
      ],
    }),
  });
  if (!resp.ok) { const e = await resp.json().catch(()=>({})); throw new Error(e?.error?.message || `Groq API error ${resp.status}`); }
  const data = await resp.json();
  return parseJSON(data.choices?.[0]?.message?.content || '');
}

async function callGemini(text, lang, apiKey, sensitivity) {
  const url    = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;
  const prompt = buildPrompt(lang, sensitivity) + `\n\nAnalyse this ${lang} news article:\n\n${text.slice(0,3000)}`;
  const resp   = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }], generationConfig: { maxOutputTokens: 900, temperature: 0.1 } }),
  });
  if (!resp.ok) { const e = await resp.json().catch(()=>({})); throw new Error(e?.error?.message || `Gemini API error ${resp.status}`); }
  const data = await resp.json();
  return parseJSON(data?.candidates?.[0]?.content?.parts?.[0]?.text || '');
}

async function callAnthropic(text, lang, apiKey, sensitivity) {
  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json', 'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514', max_tokens: 900,
      system: buildPrompt(lang, sensitivity),
      messages: [{ role: 'user', content: `Analyse this ${lang} news article:\n\n${text.slice(0,3000)}` }],
    }),
  });
  if (!resp.ok) { const e = await resp.json().catch(()=>({})); throw new Error(e?.error?.message || `Anthropic API error ${resp.status}`); }
  const data = await resp.json();
  return parseJSON(data.content?.[0]?.text || '');
}

// ── System prompt — instructs LLM to respond in native language ──
function buildPrompt(lang, sensitivity = 'balanced') {
  const guide = {
    strict:   'Flag anything with even mild fake-news indicators.',
    balanced: 'Use balanced judgment.',
    lenient:  'Only flag very obvious fake news.',
  }[sensitivity] || 'Use balanced judgment.';

  const langInstructions = {
    Hindi:    'Respond entirely in Hindi (हिंदी). All field values must be in Hindi.',
    Marathi:  'Respond entirely in Marathi (मराठी). All field values must be in Marathi.',
    Gujarati: 'Respond entirely in Gujarati (ગુજરાતી). All field values must be in Gujarati.',
    Telugu:   'Respond entirely in Telugu (తెలుగు). All field values must be in Telugu.',
    Unknown:  'Respond in English.',
  }[lang] || 'Respond in English.';

  return `You are FakeShield, a multilingual fake news detector trained on ${lang} news data using a 10-phase ML pipeline.

LANGUAGE RULE: ${langInstructions} The explanation, article_summary, detected_signals, and top_features text values must all be written in ${lang}. Only the JSON keys must remain in English.

Sensitivity: ${guide}

Fake-news signals (translate labels to ${lang}): sensational language, clickbait, unverified claims, conspiracy framing, no source cited, excessive punctuation, viral push language.
Credibility signals (translate to ${lang}): named official sources, specific dates/numbers, balanced perspective, credible news agency cited.

Respond ONLY with valid JSON, no markdown:
{
  "verdict": "FAKE"|"REAL"|"UNCERTAIN",
  "confidence": <0-100>,
  "detected_signals": {
    "fake_signals": [<signal labels in ${lang}>],
    "real_signals":  [<signal labels in ${lang}>]
  },
  "top_features": [
    {"text":"<key phrase from article in ${lang}>","direction":"fake"|"real","weight":<0-1>}
  ],
  "explanation": "<2-3 sentence explanation written in ${lang}>",
  "language_detected": "${lang}",
  "article_summary": "<one sentence in ${lang}>"
}`;
}

function parseJSON(raw) {
  try { return JSON.parse(raw.replace(/```json|```/g,'').trim()); }
  catch {
    const m = raw.match(/\{[\s\S]*\}/);
    if (m) try { return JSON.parse(m[0]); } catch {}
    return { verdict:'UNCERTAIN', confidence:50,
             detected_signals:{fake_signals:[],real_signals:[]},
             top_features:[], explanation:raw.slice(0,300),
             language_detected:'Unknown', article_summary:'' };
  }
}

// ── Render result ─────────────────────────────────────────────
function renderResult(r, lang) {
  const t       = UI[lang] || UI.Unknown;
  const verdict = (r.verdict || 'UNCERTAIN').toUpperCase();
  const conf    = Math.min(100, Math.max(0, r.confidence || 50));
  const cls     = verdict==='FAKE' ? 'fake' : verdict==='REAL' ? 'real' : 'uncertain';

  // Verdict banner — text in native language
  const banner = $('verdictBanner');
  banner.className = `verdict-banner ${cls}`;
  $('verdictIcon').textContent =
    verdict==='FAKE' ? '🚨' : verdict==='REAL' ? '✅' : '⚠️';
  $('verdictText').textContent =
    verdict==='FAKE' ? t.verdict_fake :
    verdict==='REAL' ? t.verdict_real : t.verdict_uncertain;
  $('verdictText').className = `verdict-text ${cls}`;
  $('verdictSub').textContent = r.article_summary
    ? `"${r.article_summary.slice(0,90)}${r.article_summary.length>90?'…':''}"`
    : `${r.language_detected || lang}`;

  // Engine tag
  const tagEl = $('engineTag');
  tagEl.textContent = r.engine === 'offline-ml'
    ? 'Offline ML' : ENGINE_META[activeEngine]?.label || activeEngine;
  tagEl.className = 'engine-tag';
  if (r._fallback) {
    tagEl.textContent = t.engine_tag_fallback;
    tagEl.style.background = '#fef9c3';
    tagEl.style.color = '#92400e';
  }

  // Confidence bar
  setTimeout(() => { $('confBar').style.width = conf + '%'; }, 50);
  $('confBar').className = `conf-bar ${cls}`;
  $('confPct').textContent = conf + '%';

  // Signals (already in native language from API/offline engine)
  const sigList  = $('signalsList');
  sigList.innerHTML = '';
  const fakeSigs = r.detected_signals?.fake_signals || [];
  const realSigs = r.detected_signals?.real_signals || [];
  [...fakeSigs.slice(0,4).map(s=>({tx:s,c:'red'})),
   ...realSigs.slice(0,3).map(s=>({tx:s,c:'green'}))]
  .forEach(({tx,c}) => {
    const chip = document.createElement('div');
    chip.className = `signal-chip ${c}`;
    chip.textContent = tx;
    sigList.appendChild(chip);
  });
  if (!fakeSigs.length && !realSigs.length)
    sigList.innerHTML = `<span class="no-data">${t.no_signals}</span>`;

  // Features (key phrases from article — already in native language)
  const featList = $('featureList');
  featList.innerHTML = '';
  if (r.top_features?.length) {
    r.top_features.slice(0,6).forEach(f => {
      const w   = Math.min(1, Math.max(0, f.weight||0.5));
      const dir = f.direction==='fake' ? 'pos' : 'neg';
      const dirLabel = dir==='pos' ? t.to_fake : t.to_real;
      const row = document.createElement('div');
      row.className = 'feat-row';
      row.innerHTML = `
        <span class="feat-name" title="${esc(f.text)}">${esc(String(f.text).slice(0,30))}</span>
        <div class="feat-bar-wrap"><div class="feat-bar ${dir}" style="width:${Math.round(w*100)}%"></div></div>
        <span class="feat-dir ${dir}">${esc(dirLabel)}</span>`;
      featList.appendChild(row);
    });
  } else {
    featList.innerHTML = `<span class="no-data">${t.no_features}</span>`;
  }

  // Explanation (already in native language)
  $('explanation').textContent = r.explanation || '—';

  // Language bias note (in native language)
  const detectedLang = r.language_detected || lang;
  const noteEl = $('langNote');
  if (LANG_NOTES[detectedLang]) {
    noteEl.textContent = `${t.bias_prefix} ${LANG_NOTES[detectedLang]}`;
    noteEl.classList.remove('hidden');
  } else {
    noteEl.classList.add('hidden');
  }

  $('resultCard').classList.remove('hidden');
  $('resultCard').scrollIntoView({ behavior:'smooth', block:'nearest' });
}

// ── UI helpers ────────────────────────────────────────────────
function setLoading(on) {
  isAnalysing = on;
  $('analyseBtn').disabled = on;
  $('btnSpinner').classList.toggle('hidden', !on);
  $('btnLabel').textContent = on ? 'Analysing…'
    : (activeTab==='page' ? 'Analyse Article' : 'Analyse Text');
}

function showError(msg) {
  $('errorBox').textContent = msg;
  $('errorBox').classList.remove('hidden');
}

function resetUI() {
  $('resultCard').classList.add('hidden');
  $('errorBox').classList.add('hidden');
  $('pasteArea').value = '';
  lastReport = null;
  currentLang = 'Unknown';
  // Reset labels to English
  applyUILanguage('Unknown');
}

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function copyReport() {
  if (!lastReport) return;
  const { result, lang } = lastReport;
  const t = UI[lang] || UI.Unknown;
  const lines = [
    'FakeShield Report',
    '━━━━━━━━━━━━━━━━━',
    `${t.confidence}: ${result.confidence}%  (${result.verdict})`,
    `${r?.language_detected || lang}`,
    `${result.article_summary || '—'}`,
    '',
    `${t.signals_title}: ${(result.detected_signals?.fake_signals||[]).join(', ')||'—'}`,
    `${(result.detected_signals?.real_signals||[]).join(', ')||'—'}`,
    '',
    `${t.explanation_title}: ${result.explanation}`,
    '',
    'FakeShield — Multilingual Fake News Detector',
  ];
  await navigator.clipboard.writeText(lines.join('\n'));
  const btn = $('copyBtn');
  const orig = btn.textContent;
  btn.textContent = '✅';
  setTimeout(() => { btn.textContent = orig; }, 1800);
}
