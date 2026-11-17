const ts = require('../ui/node_modules/typescript');
const fs = require('fs');
const Module = require('module');
const assert = require('assert/strict');
let requests = 0;
const load = Module._load;
Module._load = function (request, parent, ...args) {
  if (request === '../api/client' && parent.filename.endsWith('/utils/pdfSearch.ts')) {
    return { api: { get: async (url, options) => {
      requests++;
      const numbers = options.params.numbers.split(',').map(Number);
      return { data: { pages: numbers.map(page), failed: [] } };
    } } };
  }
  return load.call(this, request, parent, ...args);
};
require.extensions['.ts'] = (module, file) => module._compile(ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, file);
const { searchPages, readSearchPages } = require('../ui/src/utils/pdfSearch.ts');
const rect = { x: .1, y: .2, w: .1, h: .02 };
function page(number) { return { page: number, source: 'text', needs_ocr: false, items: [{ text: `Entry ${number}`, rect }] }; }
function baseline(pages, query) {
  const normalize = value => value.normalize('NFKC').toLowerCase().replace(/\s+/g, ' ');
  const needle = normalize(query).trim();
  if (!needle) return [];
  const matches = [];
  for (const page of [...pages].sort((a, b) => a.page - b.page)) {
    let text = ''; const spans = [];
    for (const item of page.items) { const value = normalize(item.text); if (!value) continue; const start = text.length; text += value + ' '; spans.push({ start, end: text.length - 1, rect: item.rect }); }
    let start = 0;
    while ((start = text.indexOf(needle, start)) >= 0) { const end = start + needle.length; const rects = spans.filter(span => span.start < end && span.end > start).map(span => span.rect); if (rects.length) matches.push({ page: page.page, offset: start, rects }); start = end; if (matches.length === 5000) return matches; }
  }
  return matches;
}
(async () => {
  const words = ['Alpha', 'β', 'ﬁ', 'ＡＢ', 'x', '', '  multi\nline', '123.4'];
  const pages = Array.from({ length: 40 }, (_, number) => ({ ...page(number + 1), items: Array.from({ length: 700 }, (_, item) => ({ text: words[(number * 7 + item * 3) % words.length], rect: { ...rect, x: item / 1000 } })) }));
  for (const query of ['alpha', 'β', 'fi', 'ab', 'x', 'multi line', '123.4 alpha', 'missing', ' ', 'β fi']) {
    assert.deepEqual(searchPages(pages, query), baseline(pages, query));
  }
  const large = Array.from({ length: 300 }, (_, number) => ({ ...page(number + 1), items: Array.from({ length: 1000 }, (_, item) => ({ text: `token${item}`, rect })) }));
  const start = performance.now(); searchPages(large, 'not-present'); const coldMs = performance.now() - start;
  const warmStart = performance.now(); searchPages(large, 'not-present'); const warmMs = performance.now() - warmStart;
  const controller = new AbortController();
  await readSearchPages('cached', [1, 2, 3], controller.signal);
  await readSearchPages('cached', [1, 2, 3], controller.signal);
  assert.equal(requests, 1);
  controller.abort();
  await assert.rejects(readSearchPages('cached', [1], controller.signal), { name: 'AbortError' });
  assert.equal(requests, 1);
  console.log(JSON.stringify({ baselineEquivalent: true, words: 300000, coldMs: Math.round(coldMs), warmMs: Math.round(warmMs), cache: true, abort: true }));
})().catch(error => { console.error(error); process.exit(1); });
