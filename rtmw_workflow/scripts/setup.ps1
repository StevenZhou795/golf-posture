param([string]$Python = 'python', [switch]$Cuda)
$ErrorActionPreference = 'Stop'
$WorkflowRoot = Split-Path $PSScriptRoot -Parent
$RepoRoot = Split-Path $WorkflowRoot -Parent
$VenvPython = Join-Path $WorkflowRoot '.venv/Scripts/python.exe'
Push-Location $RepoRoot
try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        & $Python -m venv (Join-Path $WorkflowRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Creating virtual environment failed' }
    }
    if ($Cuda) {
        $TorchBuild = & $VenvPython -c "import importlib.metadata as m; print(next((d.version for d in m.distributions() if d.metadata['Name']=='torch'), 'missing'))"
        if ($TorchBuild -ne '2.6.0+cu124') {
            # Free this environment's old package before installing: pip's
            # rollback copy otherwise doubles disk usage on this volume.
            & $VenvPython -m pip uninstall -y torch torchvision
            if ($LASTEXITCODE -ne 0) { throw 'Removing previous torch from the isolated environment failed' }
        }
        & $VenvPython -m pip install --no-deps torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
        if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    }
    & $VenvPython -m pip install -r (Join-Path $WorkflowRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    & $VenvPython (Join-Path $PSScriptRoot 'fetch_source.py')
    if ($LASTEXITCODE -ne 0) { throw 'Downloading upstream source failed' }
    $ClubDevice = if ($Cuda) { 'cuda' } else { 'cpu' }
    & $VenvPython -m rtmw_workflow self-test --club-device $ClubDevice
    if ($LASTEXITCODE -ne 0) { throw 'Workflow self-test failed' }
} finally {
    Pop-Location
}
