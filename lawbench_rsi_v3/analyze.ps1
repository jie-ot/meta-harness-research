$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$analysisPython = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\reference_examples\text_classification\.venv\Scripts\python.exe'))
& $analysisPython (Join-Path $PSScriptRoot 'external\tools\verify_outputs.py') --final
if ($LASTEXITCODE -ne 0) { throw '结果尚未完整或核验失败。' }
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'analysis\injected_results.json')) {
    & $analysisPython (Join-Path $PSScriptRoot 'external\tools\verify_outputs.py') --injected
    if ($LASTEXITCODE -ne 0) { throw '补测结果核验失败。' }
}
& $analysisPython (Join-Path $PSScriptRoot 'pilot\run_pilot.py') --stage analyze
if ($LASTEXITCODE -ne 0) { throw '基础分析失败。' }
foreach ($analysisStep in @('enrich_analysis.py','audit_candidate_constraints.py','render_final_report.py')) {
    & $analysisPython (Join-Path $PSScriptRoot ('external\tools\' + $analysisStep))
    if ($LASTEXITCODE -ne 0) { throw ('分析失败：' + $analysisStep) }
}
$analysisConfig = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'pilot_config.json') -Raw | ConvertFrom-Json
& $analysisConfig.plot_python (Join-Path $PSScriptRoot 'external\tools\render_extra_plots.py')
if ($LASTEXITCODE -ne 0) { throw '补充绘图失败。' }
Write-Output ('结果已核验并生成：' + (Join-Path $PSScriptRoot 'analysis\给导师的结论.md'))
