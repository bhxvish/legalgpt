# Starts LegalGPT for the demo: the backend (port 8000) and the frontend (port 5173), each in its own
# window, waits until the backend answers, then opens the app in the browser.
# Run: right-click this file > Run with PowerShell (or: powershell -ExecutionPolicy Bypass -File demo\start_demo.ps1)

$root = Split-Path -Parent $PSScriptRoot

Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root'; npm --prefix frontend run dev"

Write-Host 'Waiting for the backend (the first start builds the search indexes and can take a few minutes)...'
$ready = $false
for ($i = 0; $i -lt 180; $i++) {
    try {
        $r = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 2
        if ($r.status -eq 'ok') { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if ($ready) {
    Write-Host 'Backend ready. Opening http://localhost:5173'
    Start-Process 'http://localhost:5173'
} else {
    Write-Host 'The backend did not answer within 6 minutes. Check its window for errors, or play demo\LegalGPT_demo.mp4.'
}
