$doc = [IO.File]::ReadAllText('docs\AI_DIGEST.md')
$src = (Get-ChildItem src -Filter *.rs | ForEach-Object { [IO.File]::ReadAllText($_.FullName) }) -join "`n"

$ids = [regex]::Matches($doc, '`([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

# Names the digest raises on purpose, which cannot appear in src/. A healthy audit prints nothing under
# "PHANTOM", because output that is always noisy gets skimmed, and the one thing worth skimming for is a
# name the digest invented. Keep this list to three kinds: storage designs §2 rejects (Arena, slotmap,
# index_vec, swap_remove), operations planned but not built (delete) or deliberately removed
# (Scene::mesh_mut), external tooling, and ordinary words that sit inside backticks.
$allowed = @(
    'Arena', 'slotmap', 'index_vec', 'swap_remove',
    'delete', 'Scene::mesh_mut',
    'naga', 'cargo', 'powershell', 'llama', 'nvidia', 'Q4_K_M', 'q8_0', 'head_dim',
    'backticked', 'chars'
)

$miss = @()
foreach ($id in $ids) {
    $leaf = ($id -split '::')[-1]
    $leaf = $leaf -replace '\(.*$', ''
    if ($leaf.Length -lt 3) { continue }
    if ($src -match [regex]::Escape($leaf)) { continue }
    if ($allowed -contains $id) { continue }
    $miss += $id
}

'checked identifiers: ' + $ids.Count
if ($miss.Count -eq 0) {
    'PHANTOM (named in the digest, absent from src/, not on the allow-list): none'
} else {
    'PHANTOM (named in the digest, absent from src/, not on the allow-list):'
    $miss | ForEach-Object { '  - ' + $_ }
}
'--- digest size ---'
$t = [IO.File]::ReadAllText('docs\AI_DIGEST.md')
'est tokens: ' + [math]::Round($t.Length / 3.7) + '   chars: ' + $t.Length
exit [int]($miss.Count -gt 0)
