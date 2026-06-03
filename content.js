/* content.js — FakeShield content script
   Runs on every page. Listens for messages from popup.js
   and extracts clean article text. */

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'GET_PAGE_TEXT') {
    sendResponse({ text: extractArticleText() });
  }
  if (msg.type === 'GET_SELECTED_TEXT') {
    sendResponse({ text: window.getSelection()?.toString().trim() || '' });
  }
  return true;   // keep channel open for async
});

function extractArticleText() {
  // Priority selectors — most news sites use these
  const selectors = [
    'article',
    '[role="main"]',
    '.article-body',
    '.article-content',
    '.post-content',
    '.story-body',
    '.entry-content',
    '.news-body',
    '.content-body',
    '#article-body',
    '#story-body',
    'main',
  ];

  let container = null;
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && el.innerText.trim().length > 100) {
      container = el; break;
    }
  }
  if (!container) container = document.body;

  // Collect paragraphs and headings
  const nodes = container.querySelectorAll('p, h1, h2, h3, blockquote');
  const parts = Array.from(nodes)
    .map(n => n.innerText.trim())
    .filter(t => t.length > 15)
    .slice(0, 80);   // cap at 80 blocks

  // Include page title for context
  const title = document.title?.trim() || '';
  const body  = parts.join(' ');
  return (title ? title + '. ' : '') + body.slice(0, 4000);
}
