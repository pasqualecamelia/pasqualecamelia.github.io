# Deploy the verification section to pasqualecamelia.github.io
# Run from the ROOT of a clone of the site repository:
#     powershell -ExecutionPolicy Bypass -File .\deploy_verification.ps1
$ErrorActionPreference = "Stop"

if (-not (Test-Path "index.html")) {
  Write-Error "Run this from the root of the pasqualecamelia.github.io clone."; exit 1
}
if (-not (Test-Path "verification")) {
  Write-Error "Unpack QGT_site_verification_page.zip here first."; exit 1
}

# UTF-8 without BOM, so the HTML is not altered
$utf8 = New-Object System.Text.UTF8Encoding($false)

# 1. navigation link, next to QFMT, only if absent
$idx = [System.IO.File]::ReadAllText((Resolve-Path "index.html"), $utf8)
if ($idx -like '*href="verification/"*') {
  Write-Host "  nav link already present, skipped"
} else {
  $old = '    <a href="qfmt/">QFMT</a>'
  $new = '    <a href="qfmt/">QFMT</a>' + "`n" + '    <a href="verification/">Verification</a>'
  $count = ([regex]::Matches($idx, [regex]::Escape($old))).Count
  if ($count -ne 1) { Write-Error "navigation anchor found $count times, expected 1"; exit 1 }
  $idx = $idx.Replace($old, $new)
  [System.IO.File]::WriteAllText((Resolve-Path "index.html"), $idx, $utf8)
  Write-Host "  nav link added"
}

# 2. sitemap entry
if (Test-Path "sitemap.xml") {
  $sm = [System.IO.File]::ReadAllText((Resolve-Path "sitemap.xml"), $utf8)
  if ($sm -like "*verification/*") {
    Write-Host "  sitemap entry already present, skipped"
  } else {
    $entry = "  <url><loc>https://pasqualecamelia.github.io/verification/</loc><changefreq>monthly</changefreq><priority>0.7</priority></url>`n</urlset>"
    $sm = $sm.Replace("</urlset>", $entry)
    [System.IO.File]::WriteAllText((Resolve-Path "sitemap.xml"), $sm, $utf8)
    Write-Host "  sitemap entry added"
  }
}

Write-Host ""
Write-Host "== staged for commit =="
git add verification index.html
if (Test-Path "sitemap.xml") { git add sitemap.xml }
git status --short
