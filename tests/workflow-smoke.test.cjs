'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const pagePath = path.join(__dirname, '..', 'src', 'ide_system', 'frontend', 'ide_interface.html');

test('static IDE page exists and is non-empty', () => {
  assert.equal(fs.existsSync(pagePath), true, 'expected the checked-in IDE HTML entry point');
  assert.ok(fs.statSync(pagePath).size > 0, 'IDE HTML entry point must not be empty');
});

test('static IDE page has basic HTML document structure', () => {
  const html = fs.readFileSync(pagePath, 'utf8');
  assert.match(html, /<!doctype\s+html>/i);
  assert.match(html, /<html\b[^>]*\blang=/i);
  assert.match(html, /<head\b/i);
  assert.match(html, /<title>[^<]+<\/title>/i);
  assert.match(html, /<body\b/i);
  assert.match(html, /<\/html>\s*$/i);
});
