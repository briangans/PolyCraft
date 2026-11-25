$doc = [IO.File]::ReadAllText('docs\AI_DIGEST.md')
$src = (Get-ChildItem src -Filter *.rs | ForEach-Object { [IO.File]::ReadAllText($_.FullName) }) -join "`n"

$ids = [regex]::Matches($doc, '`([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

$miss = @()
foreach ($id in $ids) {
    $leaf = ($id -split '::')[-1]
    $leaf = $leaf -replace '\(.*$', ''
    if ($leaf.Length -lt 3) { continue }
    if ($src -match [regex]::Escape($leaf)) { continue }
    $miss += $id
}
'checked identifiers: ' + $ids.Count
'NOT FOUND in src/:'
if ($miss.Count -eq 0) { '  (none)' } else { $miss | ForEach-Object { '  - ' + $_ } }
'--- digest size ---'
$t = [IO.File]::ReadAllText('docs\AI_DIGEST.md')
'chars: ' + $t.Length + '   est tokens: ' + [math]::Round($t.Length / 3.7)
