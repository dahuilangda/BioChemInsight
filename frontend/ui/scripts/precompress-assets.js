// Build-time work only: serving assets never spends CPU compressing them.
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const root = path.resolve(process.argv[2] || path.join(__dirname, '..', 'dist'));
let count = 0, originalBytes = 0, compressedBytes = 0;
function visit(directory) {
  for (const item of fs.readdirSync(directory, { withFileTypes: true })) {
    const file = path.join(directory, item.name);
    if (item.isDirectory()) { visit(file); continue; }
    if (!/\.(?:js|mjs|css|svg|html|json)$/.test(item.name)) continue;
    const input = fs.readFileSync(file);
    if (input.length < 1024) continue;
    const gzip = zlib.gzipSync(input, { level: 6 });
    const brotli = zlib.brotliCompressSync(input, { params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 5 } });
    if (gzip.length < input.length) fs.writeFileSync(`${file}.gz`, gzip);
    if (brotli.length < input.length) fs.writeFileSync(`${file}.br`, brotli);
    count++; originalBytes += input.length; compressedBytes += Math.min(input.length, brotli.length);
  }
}
visit(root);
console.log(`Precompressed ${count} assets: ${(originalBytes / 1024).toFixed(0)} KiB → ${(compressedBytes / 1024).toFixed(0)} KiB (Brotli)`);
