const fs = require('fs');
const path = require('path');

const packageDir = path.dirname(require.resolve('jsme-editor/package.json'));
const outputDir = path.resolve(__dirname, '..', 'public', 'jsme');
const files = [
  '0ADE505A5718D4BE2E0EE1B7C54CC163.cache.js',
  '40BAF81124143A595056A9CCA0E9DBBA.cache.png',
  '4277561D0E87B89F4DFCCC3A712D5B19.cache.js',
  '96E40B969193BD74B8A621486920E79C.cache.js',
  'clear.cache.gif',
  'compilation-mappings.txt',
  'jsa.css',
  'jsme.devmode.js',
  'jsme.nocache.js',
];

fs.rmSync(outputDir, { recursive: true, force: true });
fs.mkdirSync(outputDir, { recursive: true });
for (const file of files) {
  fs.copyFileSync(path.join(packageDir, file), path.join(outputDir, file));
}
