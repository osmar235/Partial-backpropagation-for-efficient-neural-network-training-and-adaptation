param(
    [int]$StartSeed = 41,
    [int]$EndSeed = 49,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$script = Join-Path $repo 'src\agrls_stage6a_cifar10_augmented_schedule.py'
$data = Join-Path $repo 'data'
$results = Join-Path $repo 'results\stage6b'
New-Item -ItemType Directory -Force -Path $data, $results | Out-Null

for ($seed = $StartSeed; $seed -le $EndSeed; $seed++) {
    $rotation = ($seed - 41) % 3
    if ($rotation -eq 0) {
        $order = 'adam,loss_gated_moment_continuation_adam,adan'
    } elseif ($rotation -eq 1) {
        $order = 'loss_gated_moment_continuation_adam,adan,adam'
    } else {
        $order = 'adan,adam,loss_gated_moment_continuation_adam'
    }
    $out = Join-Path $results ("stage6b_heldout_seed{0}" -f $seed)
    if (-not $Force -and (Test-Path (Join-Path $out 'trajectory_summary.csv'))) {
        Write-Host "seed $seed already complete; skipping"
        continue
    }
    & $python -B $script `
        --dataset cifar10 --data-dir $data `
        --train-subset 50000 --eval-subset 10000 --batch-size 256 `
        --epochs 30 --schedule-epochs 30 --seed $seed --device cuda `
        --adam-lr 0.0003 --adan-lr 0.001 --min-lr-ratio 0.01 `
        --ema-decay 0.95 --warmup-steps 196 --algorithms $order `
        --eval-batch-size 512 --out $out
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 6B failed at seed $seed with exit code $LASTEXITCODE"
    }
}
