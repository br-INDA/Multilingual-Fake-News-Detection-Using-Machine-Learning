/* background.js — FakeShield service worker */

// Create context menu item when extension installs
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id:       'fakeshield-analyse',
    title:    '🛡️ Check with FakeShield',
    contexts: ['selection', 'page'],
  });
});

// Open popup when context menu item clicked
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'fakeshield-analyse') {
    // Store selected text so popup can pick it up
    if (info.selectionText) {
      chrome.storage.session.set({
        contextText: info.selectionText.trim(),
      });
    }
    chrome.action.openPopup();
  }
});

// Handle any unhandled errors gracefully
self.addEventListener('unhandledrejection', e => {
  console.warn('[FakeShield]', e.reason);
});
