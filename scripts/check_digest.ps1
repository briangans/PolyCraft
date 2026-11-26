$doc = [IO.File]::ReadAllText('docs\AI_DIGEST.md')
$src = (Get-ChildItem src -Filter *.rs | ForEach-Object { [IO.File]::ReadAllText($_.FullName) }) -join "`n"

$ids = [regex]::Matches($doc, '`([A-Za-z_][A-Za-z0-9_.\-/]*(?:::[A-Za-z_][A-Za-z0-9_.\-/]*)*)') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

# A backticked path that exists is documentation, not an invented API, so it is checked against the tree
# rather than the source text - `scripts/check_digest.ps1` names a file, and splitting it at the slash
# would ask whether `scripts` is a function.
$paths = $ids | Where-Object { $_ -match '[/.]' -and (Test-Path -LiteralPath $_) }

# Names the digest raises on purpose, which cannot appear in src/. A healthy audit prints nothing under
# "PHANTOM", because output that is always noisy gets skimmed, and the one thing worth skimming for is a
# name the digest invented. Keep this list to three kinds: storage designs §2 rejects (Arena, slotmap,
# index_vec, swap_remove), operations planned but not built (delete) or deliberately removed
# (Scene::mesh_mut), external tooling, and ordinary words that sit inside backticks.
$allowed = @(
    'Arena', 'slotmap', 'index_vec', 'swap_remove',
    'delete', 'Scene::mesh_mut', 'inflate',
    'naga', 'cargo', 'powershell', 'llama', 'nvidia', 'Q4_K_M', 'q8_0', 'head_dim',
    'backticked', 'chars'
)

$miss = @()
foreach ($id in $ids) {
    if ($paths -contains $id) { continue }
    # A segment of a path that exists (`scripts`, `check_digest`) is not an API claim either.
    if ($paths | Where-Object { ($_ -split '[/.]') -contains $id }) { continue }
    # Rust item paths never contain a hyphen, so a hyphenated backticked name is a crate (`egui-wgpu`),
    # a tool (`llama-server`, `nvidia-smi`) or a file (`rust-toolchain.toml`) - never a claim about an
    # item in src/, and splitting it would ask whether `server` is supposed to exist.
    if ($id -match '-') { continue }
    $leaf = (($id -split '::')[-1] -split '\.')[-1]
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
# The ceiling §0 sets. Enforced here rather than left to prose, because a limit nobody measures is
# suggestion, and the digest's own guidance is to replace rather than append - which only holds if
# something notices when appending happened anyway.
$ceiling = 12000
$t = [IO.File]::ReadAllText('docs\AI_DIGEST.md')
$tokens = [math]::Round($t.Length / 3.7)
'tokens: ' + $tokens + ' of ' + $ceiling + ' ceiling (' + [math]::Round(100 * $tokens / $ceiling) + '%)   chars: ' + $t.Length
$oversize = $tokens -gt $ceiling
if ($oversize) {
    'CEILING EXCEEDED by ' + ($tokens - $ceiling) + ' tokens. Find the rule stated twice and cut one'
    '  copy, rather than trimming the invariant that only appears once - see 0.'
}

exit [int](($miss.Count -gt 0) -or $oversize)
