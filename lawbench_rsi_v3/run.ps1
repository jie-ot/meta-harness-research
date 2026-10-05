param(
    [ValidateSet('preflight','noise','round1','round2','predict','holdout','analyze','all','injected-prepare','injected')]
    [string]$Stage = 'preflight',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:LITELLM_LOCAL_MODEL_COST_MAP = 'True'
$pilotPython = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\reference_examples\text_classification\.venv\Scripts\python.exe'))
$pilotArguments = @((Join-Path $PSScriptRoot 'pilot\run_pilot.py'), '--stage', $Stage)
if ($Resume) { $pilotArguments += '--resume' }
& $pilotPython @pilotArguments
exit $LASTEXITCODE
