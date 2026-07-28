#!/usr/bin/env bash
# Deploy the verification section to pasqualecamelia.github.io
# Run from the ROOT of a clone of the site repository.
set -euo pipefail

if [ ! -f index.html ]; then
  echo "ERROR: run this from the root of the pasqualecamelia.github.io clone"; exit 1
fi
if [ ! -d verification ]; then
  echo "ERROR: unpack QGT_site_verification_page.zip here first"; exit 1
fi

# 1. nav link, inserted next to QFMT, only if not already present
if grep -q 'href="verification/"' index.html; then
  echo "  nav link already present, skipped"
else
  python3 - <<'PY'
import io
s = io.open('index.html', encoding='utf-8').read()
old = '    <a href="qfmt/">QFMT</a>\n'
new = '    <a href="qfmt/">QFMT</a>\n    <a href="verification/">Verification</a>\n'
assert s.count(old) == 1, "navigation anchor not found exactly once"
io.open('index.html', 'w', encoding='utf-8').write(s.replace(old, new, 1))
print("  nav link added")
PY
fi

# 2. sitemap entry
if [ -f sitemap.xml ] && ! grep -q 'verification/' sitemap.xml; then
  python3 - <<'PY'
import io
s = io.open('sitemap.xml', encoding='utf-8').read()
entry = ("  <url><loc>https://pasqualecamelia.github.io/verification/</loc>"
         "<changefreq>monthly</changefreq><priority>0.7</priority></url>\n")
assert '</urlset>' in s
io.open('sitemap.xml', 'w', encoding='utf-8').write(s.replace('</urlset>', entry + '</urlset>'))
print("  sitemap entry added")
PY
fi

echo "== files that will be committed =="
git add -A verification index.html sitemap.xml 2>/dev/null || git add -A verification index.html
git status --short
