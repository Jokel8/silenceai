# PowerShell launcher for the training GUI
$venv = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    & $venv
}
python -m gui.train_gui
