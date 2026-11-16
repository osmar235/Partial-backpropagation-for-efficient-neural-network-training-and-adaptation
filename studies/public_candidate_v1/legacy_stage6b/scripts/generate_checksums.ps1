param(
    [string]$Output = "SHA256SUMS.txt"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$destination = Join-Path $repo $Output

$records = Get-ChildItem -LiteralPath $repo -File -Recurse |
    Where-Object { $_.FullName -ne $destination -and $_.FullName -notmatch '[\\/]\.git[\\/]' } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = [System.IO.Path]::GetRelativePath($repo, $_.FullName).Replace('\', '/')
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }

$records | Set-Content -LiteralPath $destination -Encoding utf8
Write-Host "Wrote $($records.Count) checksums to $destination"

